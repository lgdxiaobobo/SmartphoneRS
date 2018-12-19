package jyb.zju

import math.{pow, sqrt, abs}

object Proposed {

  def getMean(u: Int, g: Map[Int, Array[Friend]], p: Map[Int, Vector], d: Int): Vector = {
    val zeroVector = Vector(Array[Double](d))
    if (!g.contains(u)) zeroVector
    else {
      val nu = g(u).filter(f => p.contains(f.name))
      val pu = p(u)
      // val scale = if (nu.isEmpty) 1.0 else nu.map(_.score).sum
      nu.foldLeft(zeroVector)(
        (agg, x) => {
          val pv = p(x.name)
          val duv = x.score - pu.dot(pv)
          agg.add(pv, duv)
        }
      )//.mul(1.0 / scale)
    }
  }

  def main(args: Array[String]): Unit = {
    val sc = defineSpark("adding negative weighted by popular")
    // load positive feedback
    val onlyPositive = loadTrainSet(sc)
    // learn distribution in positve feedback
    val users = onlyPositive.map(_.u).distinct().collect()
    val items = onlyPositive.map(_.d).distinct().collect()
    println(
      "[INFO] only positive: %d records, %d users and %d items".format(
        onlyPositive.count(), users.length, items.length
      )
    )
    // bring confidence coefficient based on popularity
    val popularBasedWeight = onlyPositive.map(p => (p.d, p.u))
      .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      .map{case (d, us) => (d, us.size)}
    val userSum = popularBasedWeight.map(_._2).sum()
    val c0 = args(0).toDouble
    val alpha = args(1).toDouble
    val popWeight: Int => Double = n => c0 * pow(n, alpha) / pow(userSum, alpha)
    val itemWeights = popularBasedWeight.map{case (d, fd) => (d, popWeight(fd))}
      .collect().toMap
    val cd = sc.broadcast(itemWeights)

    // load essential parameters
    val sampleTimes = args(2).toInt
    val rank = args(3).toInt
    val lp = args(4).toDouble
    val lq = args(5).toDouble
    val lg = args(6).toDouble
    val maxIterTimes = args(7).toInt
    val gname = args(8)
    println("[INFO] Learning CF by d=%d, lp=%f, lq=%f for max iter %d".format(rank, lp, lq, maxIterTimes))
    println("[INFO] get graph info from %s and lg = %f".format(gname, lg))
    // load sementic location based network
    val graph = loadNumberedGraph(sc, gname)
    val gBD = sc.broadcast(graph.collect().toMap)
    val itemsBD = sc.broadcast(items)
    // mean precision and recall for 10 times training
    val meanP = new Array[Double](10)
    val meanR = new Array[Double](10)
    // 10-fold cross-validation
    for (times <- 0 until 10) {
      // random generator
      val (n1, rng) = MyRNG(42 + 17 * times).getInt
      // adding negative feedback based on sample times
      val wholeData = onlyPositive.map(p => (p.u, p.d))
        .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
        .map { case (u, used) =>
          (u, addNegative(used, itemsBD.value, sampleTimes, MyRNG(u + n1)))
        }.flatMap{case (u, ds) => ds.map(p => Sample(u, p._1, p._2))}
      // learn distribution after combining negative feedbacks
      val wholePosNum = wholeData.filter(_.xud == 1).count()
      val wholeNegNum = wholeData.filter(_.xud == 0).count()
      println("[INFO] after add negative, positive v.s. negative is %d:%d".format(wholePosNum, wholeNegNum))
      // 按照用户汇聚数据后，对正负样本进行采样
      // arrange records for every user and strainedSplit for positive and negative
      val (part1, part2) = strainedSplitWithNeg(wholeData, MyRNG(42 + times), 0.8)
      staticDataset(part1, withNeg = true)
      staticDataset(part2, withNeg = true)
      // training model
      val users = part1.map(_.u).distinct().collect()
      val items = part1.map(_.d).distinct().collect()
      val size  = part1.count() * 1.0
      val (userFactor, r1) = factorInit(users, rank, rng)
      val (itemFactor, _) = factorInit(items, rank, r1)
      // GD + line search for eta
      var fu = userFactor.toArray
      var fd = itemFactor.toArray
      // line search的参数
      val tau= 0.5
      val c  = 0.5
      val col = 1e-6
      var step = 0
      var goon = true
      while(step < maxIterTimes && goon){
        val matU = sc.broadcast(fu.toMap)
        val matD = sc.broadcast(fd.toMap)
        val error = getError(part1, matU, matD, logistic)
        val oldLoss = error.map(p => {
          val cud = if (p._1.xud == 1) 1.0 else cd.value(p._1.d)
          cud * logLoss(p._1.xud, p._2)
        }).mean()
        // val validLoss = getLoss(part2, matU, matD, cd, logLoss)
        // if (step % 5 == 0) println("[INFO] at %d step, with log-loss as %f,%f".format(step, oldLoss, validLoss))
        val gp = error.map {
          case (x, yud) =>
            val pu = matU.value(x.u)
            val qd = matD.value(x.d)
            val xud = x.xud
            val cud = if (xud == 1) 1.0 else cd.value(x.d)
            val meanP = getMean(x.u, gBD.value, matU.value, rank)
            val part1 = qd.mul(cud * (yud - xud))
            val part2 = pu.add(meanP, -1)
            val delta = part1.add(part2, lg).add(pu, lp)
            (x.u, delta)
        }.reduceByKey(_ add _)
        val gq = error.map{
          case (x, yud) =>
            val pu = matU.value(x.u)
            val qd = matD.value(x.d)
            val xud= x.xud
            val cud= if (xud == 1) 1.0 else cd.value(x.d)
            (x.d, qd.add(pu, lq, cud * (yud - xud)))
        }.reduceByKey(_ add _)
        val mp = sqrt(gp.map(_._2.getNorm2).sum())
        val mq = sqrt(gq.map(_._2.getNorm2).sum())
        val m  = -(mp + mq) / size
        val uGp = gp.map{case (u, dpu) => (u, dpu.div(mp))}
          .map{case (u, directPu) => (u, matU.value(u), directPu)}
        val uGq = gq.map{case (d, dqd) => (d, dqd.div(mq))}
          .map{case (d, directQd) => (d, matD.value(d), directQd)}
        var eta = 1.0
        var flag= true
        val t = -c * m
        while (eta > 1e-6 && flag) {
          val newP = uGp.map { case (u, pu, dpu) => (u, pu.add(dpu, -eta)) }
            .collect().toMap
          val newQ = uGq.map { case (d, qd, dqd) => (d, qd.add(dqd, -eta)) }
            .collect().toMap
          val updateLoss = getLoss(part1, newP, newQ, cd, logLoss)
          if (abs(oldLoss - updateLoss) < eta * t) eta *= tau
          else flag = false
        }
        if (eta * t < col) goon = false
        fu = uGp.map{case (u, pu, dpu) => (u, pu.add(dpu, -eta))}.collect()
        fd = uGq.map{case (d, qd, dqd) => (d, qd.add(dqd, -eta))}.collect()
        step += 1
      }
      val matUBD = sc.broadcast(fu.toMap)
      val matDBD = sc.broadcast(fd.toMap)
      val trainLoss = getLoss(part1, matUBD, matDBD, cd, logLoss)
      val validLoss = getLoss(part2, matUBD, matDBD, cd, logLoss)
      println("[INFO] at %d trys convergence, with log-loss as %f,%f".format(times, trainLoss, validLoss))
      // top-N recommendation performance evaluation
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
