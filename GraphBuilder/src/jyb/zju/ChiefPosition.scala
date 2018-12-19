package jyb.zju

import scala.math._
import java.io.{File, PrintWriter}

/*
  two-gaussian kernel density based method
  to find sementic location
*/
object ChiefPosition {
  // for given (lat, lng) in pts find neighbors within radius (e.g. h)
  def getNeighbor(pos: Position, merge: Map[String, Array[PosCntItem]], h: Double): Array[PosCntItem] = {
    val delta = h / earthR
    // find neighbors in each index-based block
    val name = getIndex(pos, delta)
    val candidates = merge(name)
    val neighbors = candidates.filter(p => haversine(pos, p.pos) <= h)
    neighbors
  }
  // partition location into index
  def partition(locs: Array[PosCntItem], h: Double): Map[String, Array[PosCntItem]] = {
    // approximate
    val delta = h / earthR
    val index = locs.map(matchIndex(delta))
    // expanding
    val idxEx = index.flatMap(expand)
    // merge by index
    val merge = idxEx.foldLeft(Map[String, Array[PosCntItem]]())(
      (agg, x) => {
        val xName = x._1
        val xPos  = x._2
        val accLst= agg.getOrElse(xName, Array[PosCntItem]()) :+ xPos
        agg + (xName -> accLst)
      }
    )
    merge
  }
  // (lat, lng) => (lat // delta, lng // delta)
  def getIndex(pos: Position, delta: Double): String = {
    val latIdx = (pos.lat.toRadians / delta).toInt
    val lngIdx = (pos.lng.toRadians / delta).toInt
    "%d#%d".format(latIdx, lngIdx)
  }
  // expand (x, y) => [(x + dx, y + dy) for dx in {-1,0,1}, dy in {-1,0,1}]
  def expand(p: (String, PosCntItem)): Array[(String, PosCntItem)] = {
    val name = p._1.split('#')
    val pos  = p._2
    val idx = name(0).toInt
    val idy = name(1).toInt
    val cross  = for (i <- -1 to 1; j <- -1 to 1) yield (i, j)
    cross.toArray.map{case (dx, dy) => ("%d#%d".format(idx + dx, idy + dy), pos)}
  }
  // given PosCntItem(Position(lat, lng), cont) => (idx, PosCntItem)
  def matchIndex(delta: Double)(p: PosCntItem): (String, PosCntItem) = {
    val pos = p.pos
    val name = getIndex(pos, delta)
    (name, p)
  }
  // format input
  def format(ps: Array[String]): PosCntItem = {
    val temp = ps(1).split('#')
    val cell = Position(temp(0).toDouble, temp(1).toDouble)
    val cont = ps(2).toInt
    PosCntItem(cell, cont)
  }
  // in local position [30,32] x [120,123]
  def localPos(p: PosCntItem): Boolean = {
    val lat = p.pos.lat
    val lng = p.pos.lng
    lat >= 30 && lat <= 32 && lng >= 120 && lng <= 123
  }
  // calculate density estimation
  def density(x: Position, pts: Array[PosCntItem], h: Double): Double = {
    val bias = log(h) + 0.5 * log(2 * Pi)
    pts.foldLeft(0.0)(
      (agg, xi) => {
        val pos = xi.pos
        val cnt = xi.cont
        val dist = haversine(x, pos) / h
        val logP = -0.5 * pow(dist, 2.0) - bias
        agg + exp(logP) * cnt
      }
    )
  }
  // utilizing mean-shift finding location with maximum density
  def meanShift(h: Double, delta: Double)(p: (String, Array[PosCntItem])): (String, Position) = {
    val user = p._1
    val locs = p._2
    // find initial points
    // cell position with max probit
    val prop0 = locs.map(p => density(p.pos, locs, h))
    val maxP0 = prop0.max
    val Position(x0, y0) = locs.apply(prop0.indexOf(maxP0)).pos
    // mean-shift on these points
    var cx = x0
    var cy = y0
    // partition
    val parts= partition(locs, h)
    // mean and shift
    var flag = true
    // common bias
    val bias = log(h) + 0.5 * log(2 * Pi)
    // shit until converge
    while (flag){
      val curr = Position(cx, cy)
      val inners = getNeighbor(curr, parts, h)
      val sumX = inners.foldLeft((0.0, 0.0))(
        (agg, xi) => {
          val pos = xi.pos
          val cnt = xi.cont
          val distance = haversine(curr, pos)
          val logP= -0.5 * pow(distance / h, 2.0) - bias
          val posX = pos.lat * exp(logP) * cnt
          val posY = pos.lng * exp(logP) * cnt
          (agg._1 + posX, agg._2 + posY)
        }
      )
      val norm = density(curr, inners, h)
      val dLat = sumX._1 / norm - cx
      val dLng = sumX._2 / norm - cy
      if (sqrt(pow(dLat, 2) + pow(dLng, 2)) < delta) flag = false
      cx = cx + dLat
      cy = cy + dLng
    }
    (user, Position(cx, cy))
  }
  // main function
  def main(args: Array[String]): Unit = {
    val sc = defineSpark("chief position")
    // get minimum distance
    // set threshold for coordinates
    val strH = args(0)
    val radius = strH.toDouble
    val crition = 0.00001
    // load position related home and work
    val home = sc.textFile("/pnrs/home_position")
      .map(_.split(',')).map(ps => (ps(0), format(ps)))
      .filter(p => localPos(p._2)).map(p => (p._1, Array(p._2)))
      .reduceByKey(_ ++ _)
    val work = sc.textFile("/pnrs/work_position")
      .map(_.split(',')).map(ps => (ps(0), format(ps)))
      .filter(p => localPos(p._2)).map(p => (p._1, Array(p._2)))
      .reduceByKey(_ ++ _)
    // sentiment path
    val sentiment = "/pnrs/sentiment_pos/"
    // Meanshift
    val homeMeanShift = home.map(meanShift(radius, crition))
    val workMeanShift = work.map(meanShift(radius, crition))
    homeMeanShift.map(p => p._1 + "|" + p._2.toString)
      .repartition(8).saveAsTextFile(sentiment + "meanshift_home_%s".format(strH))
    workMeanShift.map(p => p._1 + "|" + p._2.toString)
      .repartition(8).saveAsTextFile(sentiment + "meanshift_work_%s".format(strH))
  }
}
