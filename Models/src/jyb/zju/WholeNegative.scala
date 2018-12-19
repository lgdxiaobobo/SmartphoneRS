package jyb.zju

import math.sqrt

object WholeNegative {

  def main(args: Array[String]): Unit = {
    val sc = defineSpark("sample negative CF")
    // 仅读取有交互行为的正样本
    val onlyPositive = loadTrainSet(sc)
    // 统计只含有正阳本时样本分布
    val users = onlyPositive.map(_.u).distinct().collect()
    val items = onlyPositive.map(_.d).distinct().collect()
    println(
      "[INFO] only positive: %d records, %d users and %d items".format(
        onlyPositive.count(), users.length, items.length
      )
    )
    // 读取必须的参数
    val sampleTimes = args(0).toInt
    val rank = args(1).toInt
    val lp = args(2).toDouble
    val lq = args(3).toDouble
    val maxIterTimes = args(4).toInt
    println("[INFO] Learning CF by d=%d, lp=%f, lq=%f for max iter %d".format(rank, lp, lq, maxIterTimes))
    // 10次采样,计算模型
    val itemsBD = sc.broadcast(items)
    for (times <- 0 until 10) {
      // 设定随机数生成器
      val (n1, rng) = MyRNG(42 + times).getInt
      // 按照负样本比例扩展数据集
      val wholeData = onlyPositive.map(p => (p.u, p.d))
        .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
        .map { case (u, used) =>
          (u, addNegative(used, itemsBD.value, sampleTimes, MyRNG(u + n1)))
        }.flatMap{case (u, ds) => ds.map(p => Sample(u, p._1, p._2))}
      // 统计扩展后的正负样本情况
      val wholePosNum = wholeData.filter(_.xud == 1).count()
      val wholeNegNum = wholeData.filter(_.xud == 0).count()
      println("[INFO] after add negative, positive v.s. negative is %d:%d".format(wholePosNum, wholeNegNum))
      // 按照用户汇聚数据后，对正负样本进行采样
      val (part1, part2) = strainedSplitWithNeg(wholeData, MyRNG(42 + times), 0.8)
      staticDataset(part1, withNeg = true)
      staticDataset(part2, withNeg = true)
      // 训练模型
      val users = part1.map(_.u).distinct().collect()
      val items = part1.map(_.d).distinct().collect()
      val size  = part1.count() * 1.0
      val (userFactor, r1) = factorInit(users, rank, rng)
      val (itemFactor, _) = factorInit(items, rank, r1)
      // GD + line search for eta
      var fu = userFactor.toArray
      var fd = itemFactor.toArray
      val tau= 0.5
      val c  = 0.5
      var step = 0
      var goon = true
      while(step < maxIterTimes && goon){
        val matU = sc.broadcast(fu.toMap)
        val matD = sc.broadcast(fd.toMap)
        val error = getError(part1, matU, matD, logistic)
        val oldLoss = -error.map(p => logLoss(p._1.xud, p._2)).mean()
        val validLoss = getLoss(part2, matU, matD, logLoss)
        if (step % 5 == 0) println("[INFO] at %d step, with log-loss as %f,%f".format(step, oldLoss, validLoss))
        val gp = error.map{
          case (x, yud) =>
            val pu = matU.value(x.u)
            val qd = matD.value(x.d)
            val xud= x.xud
            (x.u, pu.add(qd, lp, yud - xud))
        }.reduceByKey(_ add _)
        val gq = error.map{
          case (x, yud) =>
            val pu = matU.value(x.u)
            val qd = matD.value(x.d)
            val xud= x.xud
            (x.d, qd.add(pu, lq, yud - xud))
        }.reduceByKey(_ add _)
        val mp = sqrt(gp.map(_._2.getNorm2).sum())
        val mq = sqrt(gq.map(_._2.getNorm2).sum())
        val m  = -(mp + mq) / size
        val uGp = gp.map{case (u, dpu) => (u, dpu.mul(1.0 / mp))}
          .map{case (u, directPu) => (u, matU.value(u), directPu)}
        val uGq = gq.map{case (d, dqd) => (d, dqd.mul(1.0 / mq))}
          .map{case (d, directQd) => (d, matD.value(d), directQd)}
        var eta = 1.0
        var flag= true
        val t = -c * m
        while (eta > 1e-6 && flag) {
          val newP = uGp.map { case (u, pu, dpu) => (u, pu.add(dpu, -eta)) }
            .collect().toMap
          val newQ = uGq.map { case (d, qd, dqd) => (d, qd.add(dqd, -eta)) }
            .collect().toMap
          val updateLoss = getLoss(part1, newP, newQ, logLoss)
          if (oldLoss - updateLoss < eta * t) eta *= tau
          else flag = false
        }
        if (eta * t < 1e-6) goon = false
        fu = uGp.map{case (u, pu, dpu) => (u, pu.add(dpu, -eta))}.collect()
        fd = uGq.map{case (d, qd, dqd) => (d, qd.add(dqd, -eta))}.collect()
        step += 1
      }
      val matUBD = sc.broadcast(fu.toMap)
      val matDBD = sc.broadcast(fd.toMap)
      val trainLoss = getLoss(part1, matUBD, matDBD, logLoss)
      val validLoss = getLoss(part2, matUBD, matDBD, logLoss)
      println("[INFO] at convergence, with log-loss as %f,%f".format(trainLoss, validLoss))
      // 对当前样本集下的数据进行topN性能检测
      val merge1 = part1.filter(_.xud == 1).map(x => (x.u, x.d))
        .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      val merge2 = part2.filter(_.xud == 1).map(x => (x.u, x.d))
        .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      val matched = merge1.join(merge2)
        .map{case (u, (known, unknown)) => (u, unknown.diff(known))}
      val changed = matched.filter(_._2.nonEmpty)
      println("[INFO] matched %d users, changed %d users".format(matched.count(), changed.count()))
      // 按照偏好估计值进行topN推荐
      val recommend = merge1.join(changed).map{case (u, (p1, _)) => (u, p1)}
        .map(recTopN(10, matUBD.value, matDBD.value))
      println("[INFO] %d users with sufficient recommendation" format recommend.count())
      val cross = changed.join(recommend).map{case (_, (t, p)) => (t, p)}
      val getF1: (Double, Double) => Double = (p, r) => 2.0 * p * r / (p + r)
      for (k <- 1 to 10){
        val crossK = cross.map{case (t, p) => (t, p.take(k).toSet)}
          .map{case (t, pk) => (t.size, t.intersect(pk).size)}
        val pk = crossK.map{case (_, tpk) => tpk * 1.0 / k}.mean()
        val rk = crossK.map{case (t, tpk) => tpk * 1.0 / t}.mean()
        val f1score = if (pk + rk == 0) 0.0 else getF1(pk, rk)
        println("%d,%f,%f,%f".format(k, pk, rk, f1score))
      }
    }
  }

  def addNegative(used: Set[Int], items: Array[Int], times: Int, rng: MyRNG):
  Array[(Int, Int)] = {
    val positive = used.toArray.map(d => (d, 1))
    val negative = items.filter(d => !used.contains(d))
      .map(d => (d, 0))
    val M = times * positive.length
    val ratio = M * 1.0 / negative.size
    times match {
      case 0 => positive
      case -1 => positive ++ negative
      case _ => {
        val sampler = negative.foldLeft((Array[(Int, Int)](), rng))(
          (agg, x) => {
            val (prop, r1) = agg._2.getDouble
            if (prop < ratio) (agg._1 :+ x, r1)
            else (agg._1, r1)
          })._1.take(M)
          positive ++ sampler
      }
    }
  }
}
