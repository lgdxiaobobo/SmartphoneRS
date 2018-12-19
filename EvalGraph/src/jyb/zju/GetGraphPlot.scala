package jyb.zju

/*
  about two works
  1. given neighborhood size k and minimum distance between neighbors d,
     plot the ratio about neighbors
  2. given neighborhood size k and minimum distance between neighbors d,
     static the common devices ratio in our location-based neighborhood
*/
import math._
import org.apache.spark.{SparkConf, SparkContext}

case class Position(lat: Double, lng: Double){
  override def toString: String = "%f#%f".format(lat, lng)
}

case class Context(uid: Int, loc1: Position, loc2: Position){
  override def toString: String = Array(uid, loc1, loc2).map(_.toString).mkString("|")
}

case class Friend(dst: Int, distance: Double){
  override def toString: String = "%d->%f".format(dst, distance)
}

object GetGraphPlot {
  val earthR = 6372.8 //km

  def haversine(pos1: Position, pos2: Position): Double = {
    val lat1 = pos1.lat.toRadians
    val lng1 = pos1.lng.toRadians
    val lat2 = pos2.lat.toRadians
    val lng2 = pos2.lng.toRadians
    val dLat = (lat1 - lat2) * .5
    val dLng = (lng1 - lng2) * .5
    val a = pow(sin(dLat), 2.0) + cos(lat1) * cos(lat2) * pow(sin(dLng), 2.0)
    val c = 2 * asin(sqrt(a))
    earthR * c
  }

  def getTwoPosition(p: String, str: String = "#"): (Position, Position) = {
    val ps = p.split(str)
    val lat1 = ps(0).toDouble
    val lng1 = ps(1).toDouble
    val lat2 = ps(2).toDouble
    val lng2 = ps(3).toDouble
    (Position(lat1, lng1), Position(lat2, lng2))
  }

  def defineFunc(d: Double): Position => (Int, Int) = {
    loc => {
      val delta = (d / earthR).toDegrees
      val idx1 = (loc.lat / delta).toInt
      val idx2 = (loc.lng / delta).toInt
      (idx1, idx2)
    }
  }

  def spreadByIndex(p: ((Int, Int), Context)):
  Seq[(String, Context)] = {
    val idx1 = p._1._1
    val idx2 = p._1._2
    val value = p._2
    val d12 = for (d1 <- -1 to 1; d2 <- -1 to 1) yield (d1, d2)
    d12.map{case (d1, d2) =>
      val ind1 = idx1 + d1
      val ind2 = idx2 + d2
      val index= "%d#%d".format(ind1, ind2)
      (index, value)
    }
  }

  def spreadByIndex(p: ((Int, Int), String, Context)):
  Seq[(String, (String, Context))] = {
    val idx1 = p._1._1
    val idx2 = p._1._2
    val value = (p._2, p._3)
    val d12 = for (d1 <- -1 to 1; d2 <- -1 to 1) yield (d1, d2)
    d12.map{case (d1, d2) =>
      val ind1 = idx1 + d1
      val ind2 = idx2 + d2
      val index= "%d#%d".format(ind1, ind2)
      (index, value)
    }
  }

  def getContext(ps: Array[String]): Context = {
    val uid = ps(0).toInt
    val (loc1, loc2) = getTwoPosition(ps(1))
    Context(uid, loc1, loc2)
  }

  def getIndex(func: Position => (Int, Int)): Context => (String, Context) = {
    p => {
      val (idx1, idx2) = func(p.loc1)
      val (idx3, idx4) = func(p.loc2)
      val index = "%d#%d#%d#%d".format(idx1, idx2, idx3, idx4)
      (index, p)
    }
  }

  def getDistance(p1: Context, p2: Context): Double = {
    val d1 = haversine(p1.loc1, p2.loc1)
    val d2 = haversine(p1.loc2, p2.loc2)
    val d12= pow(d1, 2.0) + pow(d2, 2.0)
    sqrt(d12 * 0.5)
  }

  def findFriends(src: Context, dsts: Array[Context]): (Int, Array[Friend]) = {
    val notSelf = dsts.filter(dst => !dst.uid.equals(src.uid))
    val friends = notSelf.map(dst => Friend(dst.uid, getDistance(src, dst))).sortBy(_.distance)
    (src.uid, friends)
  }

  def withinState(k: Int)(p: (Int, Array[Friend])): Array[Int] = {
    val distLst = (1 to 20).map(i => i * 0.1).toArray
    val friends = p._2
    distLst.map(d => friends.filter(_.distance <= d))
      .map(neighbors => if (neighbors.length >= k) 1 else 0)
  }

  def knn(k: Int, d: Double, friends: Array[Friend]): Array[Friend] = {
    friends.filter(_.distance <= d).take(k)
  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("static graph")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")

    val data = sc.textFile("/pnrs/source/numbered_sentiment_location")
      .map(_.split('|')).map(getContext)

    val blockify = defineFunc(2.0)

    val pos1Spread = data.map(p => (blockify(p.loc1), p)).flatMap(spreadByIndex)
    val pos2Spread = pos1Spread.map{case (ind1, p) => (blockify(p.loc2), ind1, p)}.flatMap(spreadByIndex)
    val candidates = pos2Spread.map{case (ind2, (ind1, p)) => (ind1 + "#" + ind2, p)}
      .aggregateByKey(Array[Context]())(_ :+ _, _ ++ _)
    candidates.take(1).foreach{
      case (index, nodes) =>
        println(index)
        println(nodes.map(_.toString).mkString(","))
    }
    println(candidates.count())

    val userWithIndex = data.map(getIndex(blockify))
    userWithIndex.take(2).foreach{
      case (index, p) =>
        println(index)
        println(p.toString)
    }
    println(userWithIndex.count())

    val neighbors = userWithIndex.join(candidates).map{case (_, (src, dsts)) => findFriends(src, dsts)}
    neighbors.take(1).foreach{
      case (uid, friends) =>
        println(uid)
        println(friends.map(_.toString).mkString("|"))
    }

    // for (k <- 1 to 10){
    //   println("get percentage of users with at least %d friends".format(k))
    //   val within = neighbors.map(withinState(k)).reduce((l1, l2) => l1.zip(l2).map(p => p._1 + p._2))
    //   println(within.map(_.toString).mkString("|"))
    // }

    for (k <- 1 to 10) {
      for (d <- 1 to 20) {
        val filename = "/pnrs/numbered_graph/cover_500_%d_%d".format(k, d * 100)
        println("output graph to " + filename)
        val graph = neighbors.map { case (src, friends) => (src, knn(k, d * 0.1, friends)) }
        graph.flatMap { case (src, friends) => friends.map(dst => (src, dst)) }
          .map { case (p, q) => "%d|%d|%f".format(p, q.dst, q.distance) }
          .repartition(8).saveAsTextFile(filename)
      }
    }
  }
}
