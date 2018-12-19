package jyb

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD

import math._

// definination about basic function
package object zju {

  case class PosCntItem(pos: Position, cont: Int)

  case class Edge(src: Int, dst: Int, dist: Double)

  case class Position(lat: Double, lng: Double){
    override def toString: String = "%f#%f".format(lat, lng)
    def distance(other: Position): Double = haversine(this, other)
  }

  case class PosItem(user: String, hour: Int, wDay: Int, second: Int, pos: Position){
    override def toString: String = "%s,%s".format(user, pos.toString)
  }

  def defineSpark(name: String): SparkContext = {
    val conf = new SparkConf().setAppName(name)
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    sc
  }

  def getBlockSize(D: Double): Double = {
    sqrt(2) * D / earthR
  }
  // Radius for earth
  val earthR = 6372800 // meter
  // Haversine Formula
  def haversine(pos1: Position, pos2: Position): Double = {
    val lat1 = pos1.lat.toRadians
    val lng1 = pos1.lng.toRadians
    val lat2 = pos2.lat.toRadians
    val lng2 = pos2.lng.toRadians
    val dLat = lat1 - lat2
    val dLng = lng1 - lng2

    val a = pow(sin(dLat * 0.5), 2.0) + cos(lat1) * cos(lat2) * pow(sin(dLng * 0.5), 2.0)
    val c = 2 * asin(sqrt(a))
    earthR * c
  }

  def loadPositionDict(sc: SparkContext): Broadcast[Map[String, Position]] = {
    val dict = sc.textFile("/config/Region_Cell_234G.csv").map(_.split(','))
      .map(ps => (ps(0), Position(ps(1).toDouble, ps(2).toDouble)))
      .collect().toMap
    sc.broadcast(dict)
  }

  def loadDeviceDict(sc: SparkContext): Broadcast[Map[String, String]] = {
    val dict = sc.textFile("/config/dim_terminal_total").map(_.split('|'))
      .filter(ps => ps(6).toLowerCase.equals("smartphone"))
      .map(ps => (ps(1), ps(2) + "#" + ps(3)))
      .collect().toMap
    sc.broadcast(dict)
  }

  def viewData(data: RDD[(String, Array[PosCntItem])], name1: String, name2: String): Unit = {
    val contStat = data.map(_._2).map(ps => ps.map(_.cont).sum)
      .map(m => (m, 1)).reduceByKey(_ + _).sortByKey(ascending = true)
      .map(p => "%d,%d".format(p._1, p._2))
    contStat.take(10).foreach(println)
    contStat.repartition(4).saveAsTextFile(name1)
    val distStat = data.map(_._2).map(_.length)
      .map(n => (n, 1)).reduceByKey(_ + _).sortByKey(ascending = true)
      .map(p => "%d,%d".format(p._1, p._2))
    distStat.take(10).foreach(println)
    distStat.repartition(4).saveAsTextFile(name2)
  }
  // soft-max function for normalization
  def softMax(edges: Array[(String, Double)], D: Double): Array[(String, Double)] = {
    val e1 = edges.map{case (p, d2) => (p, sqrt(d2) / D)}
    val e2 = e1.map{case (p, d) => (p, exp(-d))}
    val w0 = e2.map(_._2).sum
    e2.map{case (p, w) => (p, w / w0)}
  }

  def add(a: Int, b: Int): Int = a + b

}
