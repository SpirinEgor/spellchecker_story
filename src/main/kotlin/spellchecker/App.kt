package spellchecker

import ai.onnxruntime.OrtException
import java.io.File
import java.nio.file.Path
import java.util.logging.Level
import java.util.logging.Logger
import kotlin.io.path.Path
import kotlin.io.path.notExists

private val logger = Logger.getLogger("SpellChecker")

data class Config(val dicPath: Path, val affPath: Path) {
    companion object {
        fun buildDefaultConfig(): Config {
            val dicPath = Path("data").resolve("index.dic")
            val affPath = Path("data").resolve("index.aff")
            return Config(dicPath, affPath)
        }
    }
}

fun safeRetrieveOrderedCandidates(spellChecker: SpellChecker, word: String): List<Candidate>? =
    try {
        spellChecker.correct(word).sortedBy { -it.probability }
    } catch (e: OrtException) {
        logger.log(Level.WARNING, "Error with ONNX processing", e)
        null
    }

fun main(args: Array<String>) {
    if (args.size != 2) {
        println("Pass path to test data and ONNX model as arguments.")
        return
    }
    if (Path(args[0]).notExists()) {
        println("Passed path to test data is not a valid file")
        return
    }
    if (Path(args[1]).notExists()) {
        println("Passed path to ONNX model is not a valid file")
        return
    }

    val config = Config.buildDefaultConfig()
    val spellChecker = SpellChecker(config.dicPath, config.affPath, args[1])
    val results = File(args[0]).readLines().map { line ->
        val (misspell, correct) = line.split("\t").map { it.trim() }
        val orderedCandidates = safeRetrieveOrderedCandidates(spellChecker, misspell) ?: return@map null
        val isTop1 = orderedCandidates[0].word == correct
        val isTop5 = orderedCandidates.take(5).map { it.word }.contains(correct)
        Pair(isTop1, isTop5)
    }
    val total = results.size
    val skipped = total - results.filterNotNull().size
    val acc1 = results.filterNotNull().count { it.first }.toFloat() / total * 100
    val acc5 = results.filterNotNull().count { it.second }.toFloat() / total * 100
    println(
        "Evaluate $total examples, skipped $skipped.\n" +
                "Accuracy@1: ${acc1.round(2)}, accuracy@5: ${acc5.round(2)}."
    )
}
