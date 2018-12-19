package jyb.zju

/*
  item2item recommendation
  compare items by their users
  then recommend users with similar items
  ps. make up items with popular items for lackness
*/
object ItemCF {

  def expand(items: Array[(Int, Double)], user: Int): Seq[(Int, Int, Int, Double)] = {
    items.flatMap{
      case (i, wi) =>
        items.filter(_._1 != i)
          .map{case (j, wj) => (user, i, j, wi * wj)}
    }
  }

  def concate(i: Int, j: Int): String = "%d#%d".format(i, j)
  def deconcate(str: String): Array[Int] = str.split('#').map(_ toInt)

  def mergeLst(lst: Array[(Int, Double)]): Array[(Int, Double)] = {
    lst.foldLeft(Map[Int, Double]())(
      (agg, x) =>
        agg + (x._1 -> (agg.getOrElse(x._1, 0.0) + x._2))
    ).toArray.sortBy(-_._2)
  }

  def main(args: Array[String]): Unit = {

    val sc = defineSpark("item-based CF")
    val train = loadTrainSet(sc)
    val alpha = args(0).toDouble
    for (times <- 0 until 10) {
      println("============stage %d==============" format times)
      val rng = MyRNG(42 + times)
      val (part1, part2) = strainedSplit(train, rng, alpha)
      println("[INFO] static on train-set")
      staticDataset(part1)
      println("[INFO] static on valid-set")
      staticDataset(part2)
      // popular items
      val popItems = part1.map(_.d).map(d => (d, 1))
        .reduceByKey(add).sortBy(-_._2).map(_._1).take(20)
      println("[INFO] most popular 20 device is %s".format(popItems.mkString("|")))
      val popBD = sc.broadcast(popItems)
      // obtain scores
      // Score_{item} = \frac{1}{1+\sqrt{the number of users}}
      // Score_{user} = \frac{1}\log{1+{the number of used devices}}
      val itemScore = part1.map(_.d).map(d => (d, 1))
        .reduceByKey(add).map{case (d, c) => (d, sqrtScore(c))}
      val userScore = part1.map(_.u).map(u => (u, 1))
        .reduceByKey(add).map{case (u, c) => (u, logScore(c))}
      val itemPairs = part1.map(p => (p.u, p.d))
        .map{case (u, d) => (d, u)}.join(itemScore)
        .map{case (d, (u, wd)) => (u, (d, wd))}
        .aggregateByKey(Array[(Int, Double)]())(_ :+ _, _ ++ _)
        .flatMap{case (user, items) => expand(items, user)}
      val itemSimScore = itemPairs.map{
        case (u, i, j, wij) => (u, (i, j, wij))
      }.join(userScore).map{
        case (_, ((i, j, wij), wu)) =>
          (i, j, wij * wu)
      }
      // obtain the top-10 items comparing with similarities
      val itemTop10 = itemSimScore.map{case (i, j, sij) => (concate(i, j), sij)}
        .reduceByKey(add).map{case (k, v) => (deconcate(k), v)}
        .map{case (ks, v) => (ks(0), (ks(1), v))}
        .aggregateByKey(Array[(Int, Double)]())(_ :+ _, _ ++ _)
        .map{case (i, js) => (i, js.sortBy(-_._2).take(10))}
      // recommend to users with similar items
      // for users with less than N recommended items, 
      // we take popular alternatively
      val basedItem = part1.map(p => (p.u, p.d))
        .map{case (u, d) => (d, u)}.join(itemTop10)
        .map{case (_, (u, top10)) => (u, top10)}
        .reduceByKey(_ ++ _).map{case (u, top10s) => (u, mergeLst(top10s))}
      val recommendation = part1.map(p => (p.u, p.d))
        .aggregateByKey(Array[Int]())(_ :+ _, _ ++ _)
        .leftOuterJoin(basedItem).map{
        case (u, (used, topN)) => {
          val usedSet = used.toSet
          val unused = popBD.value.filter(d => !usedSet.contains(d))
          topN match {
            case None => (u, unused.take(10))
            case Some(lst: Array[(Int, Double)]) =>
              val recommend = lst.filter(p => !usedSet.contains(p._1))
              val N0 = if (recommend.length > 10) 10 else recommend.length
              val N1 = 10 - N0
              (u, recommend.take(N0).map(_._1) ++ unused.take(N1))
          }
        }
      }
      // evalulate result in test part
      val merge1 = part1.map(p => (p.u, p.d))
        .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      val merge2 = part2.map(p => (p.u, p.d))
        .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      val merged = merge1.join(merge2)
      println("[INFO] %d users in both train and valid" format merged.count())
      val changed= merged.map{case (u, (p1, p2)) => (u, p2.diff(p1))}
        .filter(_._2.nonEmpty)
      println("[INFO] %d users change devices" format changed.count())
      val perform = changed.join(recommendation).map{case (_, (t, p)) => (t, p)}
      println("[INFO] %d users with recommendation" format perform.count())
      println("[INFO] the performance while recommendation top-n devices")
      for (k <- 1 to 10){
        val cross = perform.map{case (t, p) => (t, p.take(k).toSet)}
        val mpk = cross.map{case (t, pk) => t.intersect(pk).size * 1.0 / k}.mean()
        val mrk = cross.map{case (t, pk) => t.intersect(pk).size * 1.0 / t.size}.mean()
        val mf1 = if (mpk + mrk == 0) 0.0 else 2 * mpk * mrk / (mpk + mrk)
        println("top-%d,%f,%f,%f".format(k, mpk, mrk, mf1))
      }
    }

  }

}
