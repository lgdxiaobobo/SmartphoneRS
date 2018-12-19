package jyb.zju

/*
  Hierarchical Neighborhood Discovery method
*/

object GetPosGraph {
  // format locations
  def localFormat(str: Char)(p: String): (String, Position) = {
    val ps = p.split(str)
    val user = ps(0)
    val temp = ps(1).split('#').map(_.toDouble)
    (user, Position(temp(0), temp(1)))
  }
  // get block-id by locations (longitude and latitude)
  def getBlockID(p: Position, H: Double): String = {
    val lat = p.lat.toRadians
    val lng = p.lng.toRadians
    val idx = lat / H
    val idy = lng / H
    "%d#%d".format(idx.toInt, idy.toInt)
  }
  // expand users to adjoint blocks
  def localExpand(p: (String, (String, Position, Position))): Seq[(String, (String, Position, Position))] = {
    val index = p._1
    val temp = index.split('#')
    val idx = temp(0).toInt
    val idy = temp(1).toInt
    val lst = for (i <- -1 to 1; j <- -1 to 1) yield (i, j)
    lst.map{
      case (dx, dy) =>
        val key = "%d#%d".format(idx + dx, idy + dy)
        (key, p._2)
    }
  }
  // find neighbors for each user by comparing distances
  def knn(refer: (String, Position, Position), candidates: Array[(String, Position, Position)], N: Int): (String, Array[(String, Double)]) = {
    val user = refer._1
    val hPos = refer._2
    val wPos = refer._3
    val distance = candidates.filter(p => !p._1.equals(user))
      .map{
        case (v, h, w) =>
          val dH = haversine(hPos, h)
          val dW = haversine(wPos, w)
          (v, dH * dH + dW * dW)
      }
    val neighbors = distance.sortBy(_._2).take(N)
    (user, neighbors)
  }
  // rid off edges with more than D distance
  def getValidEdge(D: Double)(p: (String, Array[(String, Double)])): (String, Array[(String, Double)]) = {
    val src = p._1
    val D2 = D * D * 2
    val edges = p._2.filter(_._2 <= D2)
    edges.length match {
      case 0 => (src, Array[(String, Double)]())
      case _ => (src, softMax(edges, D))
    }
  }

  def main(args: Array[String]): Unit = {
    val sc = defineSpark("get position similar graph")
    // load sementaic locations for residence and workplace
    val srcPath = args(0)
    val home = sc.textFile(srcPath.format("home"))
      .map(localFormat('|'))
    val work = sc.textFile(srcPath.format("work"))
      .map(localFormat('|'))
    println("[INFO] finding %d users with home position".format(home.count()))
    println("[INFO] finding %d users with work position".format(work.count()))
    // join dataset about users with specific residence and workplace
    val both = home.join(work).map{case (user, (home, work)) => (user, home, work)}
    println("[INFO] finding %d users with both home and work position".format(both.count()))
    // input the maximum distance D and maximum neighors N
    val D = args(1).toFloat
    val N = args(2).toInt
    println("[INFO] find top-%d neighbors within %f".format(N, D))
    val dstPath = args(3)
    println(dstPath)
    // approximate the distance in radius
    val H = getBlockSize(D)
    /*
      Hierarchical Neighborhood Discovery method
      1. divide users by their residence firstly and then workplace into blocks
      2. spread users to adjoint block
      3. find no more than N neighbors in his/her corresponding augmented block for every user
    */
    // divide users into blocks by their residence (considering block constraint)
    val partitionByHome = both.map{case (user, hPos, wPos) => (getBlockID(hPos, H), (user, hPos, wPos))}
    // expand residence-based block adjointly
    val expandWithHome = partitionByHome.flatMap(localExpand)
    // obtain residence-based augmented blocks
    val mergeWithHome = expandWithHome.map{case (blockID, blockValue) => (blockID, Array(blockValue))}
      .reduceByKey(_ ++ _)
    println("[INFO] after block by home position, there are %d blocks".format(mergeWithHome.count()))
    // divide users into sub-blocks by their workplace (considering constraint in residence-based blocks)
    val partitionByWork = mergeWithHome.flatMap{
      case (hID, items) =>
        items.map{case (user, hPos, wPos) => (getBlockID(wPos, H), (hID + "|" + user, hPos, wPos))}
    }
    // expand residence-workplace-based blocks adjointly
    val expandWithWork = partitionByWork.flatMap(localExpand)
    // obtain final augmented blocks
    val mergeWithWork = expandWithWork.map{
      case (wID, (temp, hPos, wPos)) =>
        val temp1 = temp.split('|')
        val hID = temp1(0)
        val user = temp1(1)
        val blockID = hID + "|" + wID
        val blockValue = (user, hPos, wPos)
        (blockID, Array(blockValue))
    }.reduceByKey(_ ++ _)
    println("[INFO] after block by work position, there are %d blocks".format(mergeWithWork.count()))
    // learn users' specific block-id
    // and augmented block by his/her residence and workplace
    val labeled = both.map{
      case (user, hPos, wPos) =>
        (getBlockID(hPos, H) + "|" + getBlockID(wPos, H), (user, hPos, wPos))
    }
    // labeled.take(10).foreach(println)
    // both.take(10).map{case (_, hPos, wPos) => getBlockID(hPos, H) + "|" + getBlockID(wPos, H)}
    //     .foreach(println)
    // mergeWithWork.map(_._1).take(10).foreach(println)
    val findNeighbors = labeled.join(mergeWithWork)
      .map{case (_, (refer, candidates)) => knn(refer, candidates, N)}
    // findNeighbors.take(10)
    //   .foreach{case (src, dsts) => dsts.foreach(p => println("%s,%s,%f".format(src, p._1, p._2)))}
    // rid-off edges with more than D distance
    // soft-max about their min-max distance for similarity scores
    val validEdges = findNeighbors.map(getValidEdge(D))
      .flatMap{case (src, dsts) => dsts.map{case (dst, sim) => (src, dst, sim)}}
    validEdges.take(10).foreach(println)
    validEdges.map{case (src, dst, score) => "%s,%s,%f".format(src, dst, score)}
      .repartition(16).saveAsTextFile(dstPath)
  }
}
