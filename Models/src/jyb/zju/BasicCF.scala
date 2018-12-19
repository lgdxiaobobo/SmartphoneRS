package jyb.zju

import math.sqrt

// Basic CF
// X = UTP
// U for users' latent factors
// P for items' latent factors
// X for observed feedback
object BasicCF {

  def main(args: Array[String]): Unit = {
    val sc = defineSpark("basic CF")
    val train = loadTrainSet(sc)

    val tSize = train.count()
    val users = train.map(_.u).distinct()
    val uNum  = users.count()
    val items = train.map(_.d).distinct()
    val iNum  = items.count()
    println("[INFO] at train set, there are %d usage records, %d users and %d devices".format(tSize, uNum, iNum))

    val rank = args(0).toInt
    val lp = args(1).toDouble
    val lq = args(2).toDouble
    val maxIterTimes = args(3).toInt
    println("[INFO] Learning CF by d=%d, lp=%f, lq=%f for max iter %d".format(rank, lp, lq, maxIterTimes))

    for (times <- 0 until 10){
      val rng = MyRNG(42 + times)
      val (part1, part2) = strainedSplit(train, rng, 0.8)
      println("[INFO] static on train-set")
      staticDataset(part1)
      println("[INFO] static on valid-set")
      staticDataset(part2)

      val users = part1.map(_.u).distinct().collect()
      val items = part1.map(_.d).distinct().collect()
      val size  = part1.count() * 1.0
      val (userFactor, r1) = factorInit(users, rank, rng)
      val (itemFactor, r2) = factorInit(items, rank, r1)
      // // initial loss
      // val loss1 = getLoss(part1, userFactor, itemFactor, logLoss)
      // val loss2 = getLoss(part2, userFactor, itemFactor, logLoss)
      // GD + line search for eat
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
          // println("%f,%f,%f" format(eta, updateLoss, oldLoss - eta * t))
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

      val merge1 = part1.map(p => (p.u, p.d))
        .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      val merge2 = part2.map(p => (p.u, p.d))
        .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      val matched = merge1.join(merge2)
        .map{case (u, (known, unknown)) => (u, unknown.diff(known))}
      val changed = matched.filter(_._2.nonEmpty)
      println("[INFO] matched %d users, changed %d users".format(matched.count(), changed.count()))

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

}
