package jyb.zju

import math.{log, sqrt}

// Ranked sample for BPR
case class MyRank(user: Int, used: Int, unused: Int){
  def items: Array[Int] = Array(used, unused)
  override def toString: String = "%d: %d > %d" format(user, used, unused)
}

object BPR {
  // Under the assumption about Bayesian Personal Ranking
  // the preference about used items is higher than unused
  def main(args: Array[String]): Unit = {
    val sc = defineSpark("my bpr")
    // load positive feedback
    val onlyPositive = loadTrainSet(sc)
    // learn distribution for positive feedback
    val users = onlyPositive.map(_.u).distinct().collect()
    val items = onlyPositive.map(_.d).distinct().collect()
    println(
      "[INFO] only positive: %d records, %d users and %d items".format(
        onlyPositive.count(), users.length, items.length
      )
    )
    // load negative sampling ratio
    val sampleTimes = args(0).toInt
    println("[INFO] for each positive sample, match %d negative".format(sampleTimes))
    // load involved paramters
    val rank = args(1).toInt
    val lp = args(2).toDouble
    val lq = args(3).toDouble
    val maxIterTimes = args(4).toInt
    println("[INFO] Learning BPR by d=%d, lp=%f, lq=%f for max iter %d".format(rank, lp, lq, maxIterTimes))
    val itemsBD = sc.broadcast(items)
    // mean precision and recall for 10-fold cross-validation
    val meanP = new Array[Double](10)
    val meanR = new Array[Double](10)
    for (times <- 0 until 10){
      // random generator for adding negative samples
      val (n1, rng) = MyRNG(42 + times).getInt
      // expand samples by sampling ratio
      val wholeData = onlyPositive.map(p => (p.u, p.d))
        .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
        .map { case (u, used) =>
          (u, addNegative(used, itemsBD.value, sampleTimes, MyRNG(u + n1)))
        }.flatMap{case (u, ds) => ds.map(p => Sample(u, p._1, p._2))}
      // learn distribution on augmented dataset
      val wholePosNum = wholeData.filter(_.xud == 1).count()
      val wholeNegNum = wholeData.filter(_.xud == 0).count()
      println("[INFO] after add negative, positive v.s. negative is %d:%d".format(wholePosNum, wholeNegNum))
      // accumulate records and strainedSplit
      val (data1, data2) = strainedSplitWithNeg(wholeData, MyRNG(42 + times), 0.8)
      staticDataset(data1, withNeg = true)
      staticDataset(data2, withNeg = true)
      val part1 = getRankSample(data1)
      val part2 = getRankSample(data2)
      part1.take(5).foreach(println)
      println("[INFO] %d train-set, %d valid-set" format(part1.count(), part2.count()))
      // training model alternatively
      val users = part1.map(_.user).distinct().collect()
      val items = part1.flatMap(_.items).distinct().collect()
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
        val error = getRankError(part1, matU, matD, logistic)
        val oldLoss = error.map(p => -log(p._2)).mean()
        // val validLoss = getRankLoss(part2, matU, matD)
        // if (step % 5 == 0) println("[INFO] at %d step, with log-loss as %f,%f".format(step, oldLoss, validLoss))
        val gp = error.map{
          case (x, yud) =>
            val pu = matU.value(x.user)
            val qd1 = matD.value(x.used)
            val qd2 = matD.value(x.unused)
            val dq = qd1.add(qd2, -1.0)
            (x.user, pu.add(dq, lp, yud - 1))
        }.reduceByKey(_ add _)
        val gq = error.flatMap{
          case (x, yud) =>
            val pu = matU.value(x.user)
            val qd1 = matD.value(x.used)
            val qd2 = matD.value(x.unused)
            val gq1 = qd1.add(pu, lq, yud - 1)
            val gq2 = qd2.add(pu, lq, 1 - yud)
            Array((x.used, gq1), (x.unused, gq2))
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
          val updateLoss = getRankLoss(part1, newP, newQ)
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
      val trainLoss = getRankLoss(part1, matUBD, matDBD)
      val validLoss = getRankLoss(part2, matUBD, matDBD)
      println("[INFO] at convergence, with log-loss as %f,%f".format(trainLoss, validLoss))
      // 对当前样本集下的数据进行topN性能检测
      val merge1 = part1.map(x => (x.user, x.used))
        .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      val merge2 = part2.map(x => (x.user, x.used))
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
      for (k <- 1 to 10){
        val crossK = cross.map{case (t, p) => (t, p.take(k).toSet)}
          .map{case (t, pk) => (t.size, t.intersect(pk).size)}
        val pk = crossK.map{case (_, tpk) => tpk * 1.0 / k}.mean()
        val rk = crossK.map{case (t, tpk) => tpk * 1.0 / t}.mean()
        // val f1score = if (pk + rk == 0) 0.0 else getF1(pk, rk)
        // println("%d,%f,%f,%f".format(k, pk, rk, f1score))
        meanP(k-1) += pk * 0.1
        meanR(k-1) += rk * 0.1
      }
    }
    println("[INFO] mean perfomance on graph")
    for (k <- 1 to 10){
      val f1Score = 2.0 * meanP(k-1) * meanR(k-1) / (meanP(k-1) + meanR(k-1))
      println("%d,%f,%f,%f".format(k, meanP(k-1), meanR(k-1), f1Score))
    }
  }

}
