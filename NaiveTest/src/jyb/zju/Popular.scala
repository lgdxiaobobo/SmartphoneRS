package jyb.zju
/*
  recommend popular phones
*/
object Popular {

  def main(args: Array[String]): Unit = {
    val sc = defineSpark("recommendation only based popular")
    val train = loadTrainSet(sc)
    for (cnt <- 0 until 10){
      val rng = MyRNG(42 + cnt)
      val (part1, part2) = strainedSplit(train, rng, 0.8)
      val top10 = part1.map(x => (x.d, x.u))
        .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
        .mapValues(_.size).sortBy(-_._2).take(10)
      top10.foreach(println)
      val result = top10.map(_._1)
      val arrange1 = part1.map(p => (p.u, p.d)).aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      val arrange2 = part2.map(p => (p.u, p.d)).aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      val changed = arrange1.join(arrange2).mapValues(p => p._2.diff(p._1))
        .filter(_._2.nonEmpty).map(_._2)
      println("%d users haved changed devices".format(changed.count()))
      for (k <- 1 to 10){
        val popK = result.take(k).toSet
        val popKBD = sc.broadcast(popK)
        val precision = changed.map(c => c.intersect(popKBD.value).size * 1.0 / k).mean()
        val recall = changed.map(c => c.intersect(popKBD.value).size * 1.0 / c.size).mean()
        println("%d,%f,%f".format(k, precision, recall))
      }
    }
  }

}
