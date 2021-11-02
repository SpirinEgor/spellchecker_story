package spellchecker

import kotlin.math.round

data class Candidate(val word: String, val probability: Float) {
    override fun toString(): String = "$word (p=${(probability * 100).round(2)})"
}

fun Float.round(decimals: Int): Double {
    var multiplier = 1.0
    repeat(decimals) { multiplier *= 10 }
    return round(this * multiplier) / multiplier
}
