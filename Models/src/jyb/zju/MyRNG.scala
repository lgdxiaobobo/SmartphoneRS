package jyb.zju

import math._

// My functional RNG class
case class MyRNG(seed: Long) {

  private def nextInt: (Int, MyRNG) = {
    val seed1 = (seed * 0x5DEECE66DL + 0xBL)  & 0xFFFFFFFFFFFFL
    val r1 = MyRNG(seed1)
    val n1 = (seed1 >>> 16).toInt
    (n1, r1)
  }

  private def nextDouble: (Double, MyRNG) = {
    val (n, r) = nextInt
    val d = n / (1.0 + Int.MaxValue)
    (d, r)
  }

  def getDouble: (Double, MyRNG) = {
    val (d0, r) = nextDouble
    val d1 = (1 + d0) * 0.5
    (d1, r)
  }

  def getDouble(a: Double, b: Double): (Double, MyRNG) = {
    val (d0, r) = getDouble
    val d1 = d0 * (b - a) + a
    (d1, r)
  }

  def getInt: (Int, MyRNG) = nextInt

  def getInt(a: Int, b: Int): (Int, MyRNG) = {
    val (n0, r) = nextInt
    val n1 = abs(n0) % (b - a) + a
    (n1, r)
  }
}
