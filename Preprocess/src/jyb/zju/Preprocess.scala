package jyb.zju

import java.text.SimpleDateFormat

/* 
    basic filtering process
    1. valid date, e.g. from 20161023 to 20170215
    2. proper cell-id, matching annoymized location (longitude and latitude)
    3. proper tac, matching smart-phone
*/ 
/*
    split dataset into candidates based on proper registration period
    e.g., in China, residence before 7am or after 8pm
                    workplace from 9am to 5pm in workday (Mon to Fri)
*/
object Preprocess {

  def getWDay(sec: Int): Int = {
    val dict = Map("Mon" -> 0, "Tue" -> 1, "Wed" -> 2, "Thu" -> 3, "Fri" -> 4, "Sat" -> 5, "Sun" ->6)
    val wDay = new SimpleDateFormat("EEE").format(sec * 1000L)
    dict(wDay)
  }

  def getHour(sec: Int): Int = {
    new SimpleDateFormat("HH").format(sec * 1000L).toInt
  }

  def formatData(dct: Map[String, Position]): Array[String] => PosItem = {
    ps => {
      val second = ps(0).toInt
      val hour = getHour(second)
      val wDay = getWDay(second)
      val user = ps(1)
      val position = dct(ps(3))
      PosItem(user, hour, wDay, second, position)
    }
  }

  // candidates for residence (e.g., before 7am or after 8pm)
  def isHome(p: PosItem): Boolean = {
    p.hour < 8 || p.hour >= 20
  }
  // candidates for workplace (e.g., from 9am to 4pm in workday)
  def isWork(p: PosItem): Boolean = {
    p.wDay < 5 && p.hour >= 9 && p.hour <= 16
  }

  def getDay(sec: Int): Int = {
    new SimpleDateFormat("yyyyMMdd").format(sec * 1000L).toInt
  }

  def validDay(second: Int): Boolean = {
    val day = getDay(second)
    day <= 20170215 && day >= 20161023
  }

  def main(args: Array[String]): Unit = {
    val sc = defineSpark("preprocess")
    val locDict = loadPositionDict(sc)
    val tacDict = loadDeviceDict(sc)
    for (part <- 0 until 16) {
      println("Deal with users with last %d".format(part))
      val data = sc.textFile("/warehouse/user_data/user_with_last_%d".format(part), 128)
        .map(_.split('|')).filter(ps => locDict.value.contains(ps(3)))
        .filter(ps => tacDict.value.contains(ps(2).take(8)))
        .filter(ps => validDay(ps(0).toInt))
      println("[INFO] %d records remains".format(data.count()))
      data.map(ps => Array(ps(0), ps(1), ps(2).take(8), ps(3)))
        .map(_.mkString("|")).repartition(16)
        .saveAsTextFile("/pnrs/validData/last_idx_%d".format(part))
    }

    val data = sc.textFile("/pnrs/validData/last_idx_*").map(_.split('|'))

    val days = data.map(_.head.toInt).map(getDay).distinct().count()
    val users = data.map(_.apply(1)).distinct().count()
    val devices = data.map(_.apply(2)).map(tacDict.value.apply).distinct().count()
    val cells = data.map(_.apply(3)).distinct().count()
    println("[INFO] after ridding off invalid data, there are %d users, %d cells, %d devices and %d days".format(users, cells, devices, days))

    val formatted = data.map(formatData(locDict.value))
    val home = formatted.filter(isHome)
    val work = formatted.filter(isWork)

    home.map(_.toString).map(x => (x, 1)).reduceByKey(_ + _)
      .map{case (k, v) => "%s,%d".format(k, v)}.repartition(16)
      .saveAsTextFile("/pnrs/home_position")
    work.map(_.toString).map(x => (x, 1)).reduceByKey(_ + _)
      .map{case (k, v) => "%s,%d".format(k, v)}.repartition(16)
      .saveAsTextFile("/pnrs/work_position")
  }

}
