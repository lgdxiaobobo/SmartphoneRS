package jyb.zju

import org.apache.spark.{SparkConf, SparkContext}
/*
  compare common devices ratio among friends
  here we take the most popular 10 devices as baselines
  output the matching ratio 
  while varying knn from 1 to 10 and distance from 100 to 20000
*/
object CompareDevice {

  def cover(d1: Set[Int], d2: Set[Int]): Double = {
    val d12 = d1.intersect(d2)
    // d12.size * 1.0 / d1.size
    if (d12.nonEmpty) 1.0 else 0.0
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("compare device ratio among friends")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    val userWithDevice = sc.textFile("/pnrs/source/numbered_dataset")
      .map(_.split('|')).map(ps => (ps(1).toInt, ps(2).toInt))
      .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
    println(userWithDevice.count())
    println(userWithDevice.map(_._2).map(_.size).sum())

    val popular = userWithDevice.flatMap{_._2}.map(x => (x, 1)).reduceByKey(_ + _).sortBy(-_._2).take(10).map(_._1)

    for (knn <- 1 to 10){
      val popLst = sc.broadcast(popular.take(knn).toSet)
      val dists = (1 to 20).map(_ * 100).toArray
      val scores = dists.map(dist => {
        val filename = "/pnrs/numbered_graph/cover_500_%d_%d".format(knn, dist)
        val graph = sc.textFile(filename).map(_.split('|'))
        val matchDevice = graph.map(ps => (ps(0).toInt, ps(1).toInt))
          .join(userWithDevice).map{case (i, (j, di)) => (j, (i, di))}
          .join(userWithDevice).map{case (_, ((i, di), dj)) => (i, cover(di, dj))}
        matchDevice.aggregateByKey(Array[Double]())(_ :+ _, _ ++ _)
          .map(_._2).map(lst => lst.sum / lst.length).mean()
      })
      val popScore = userWithDevice.map(_._2).map(p => popLst.value.intersect(p).size * 0.1).mean()
      val result = (knn +: scores) :+ popScore
      println(result.map(_.toString).mkString("|"))
    }
  }

}
