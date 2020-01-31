package org.apache.spark.ml.outlier.utils

import java.util.{Comparator, PriorityQueue}

import scala.collection.mutable.ArrayBuffer

/**
  * A priority queue that maintains a fixed size k, ordered by the value in the tuple elements.
  * It is used for k-NN search.
  * The value in the tuple is supposed to be the distance between a data point and its neighbor.
  * Special cases considered: identical data points; neighbors with the same distances
  *
  * @param k the number of neighbors, as in k-nearest-neighbors
  * */
class KNNPriorityQueue[K](val k: Int) extends Serializable {

  //maxHeap
  val queue =
    new java.util.PriorityQueue[(K, Double)](new Comparator[(K, Double)]() {
      override def compare(o1: (K, Double), o2: (K, Double)): Int =
        if (o2._2 - o1._2 > 0) 1 else if ((o2._2 - o1._2).equals(0)) 0 else -1
    })
  var kCount = 0

  // whether zero distance exists, i.e., identical data points exist.
  var zeroExists = false

  def offer(element: (K, Double)): this.type = {
    if (element._2.equals(0d)) {
      //println(s"${element.toString()} equals 0!")
      queue.offer(element)
      if (!zeroExists) {
        zeroExists = true
        if (kCount.equals(k)) {
          queue.poll()
        } else {
          kCount += 1
        }
      }
    } else if (queue.isEmpty) {
      queue.offer(element)
      kCount += 1
    } else {
      val (topK, topV) = queue.peek()
      if (topV.equals(element._2)) {
        if (kCount < k) {
          queue.offer(element)
          kCount += 1
        }
      } else if (topV > element._2) {
        if (kCount < k) {
          queue.offer(element)
          kCount += 1
        } else {
          queue.poll()
          queue.offer(element)
        }
      } else if ((topV < element._2) && (kCount < k)) {
        queue.offer(element)
        kCount += 1
      }
    }
    this
  }

  def dequeueALl(): Array[(K, Double)] = {
    val buffer = new ArrayBuffer[(K, Double)](queue.size)
    while (!queue.isEmpty) {
      buffer += queue.poll()
    }
    buffer.toArray.reverse
  }
}
