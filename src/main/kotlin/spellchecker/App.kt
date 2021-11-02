package spellchecker

import ai.onnxruntime.OrtException
import java.io.File
import java.nio.file.Path
import java.util.logging.Level
import java.util.logging.Logger
import kotlin.io.path.Path
import kotlin.io.path.notExists

private val logger = Logger.getLogger("SpellChecker")

data class Config(val dicPath: Path, val affPath: Path, val modelPath: Path) {
    companion object {
        fun buildDefaultConfig(): Config {
            val dicPath = Path("data").resolve("index.dic")
            val affPath = Path("data").resolve("index.aff")
            val modelPath = Path("checkpoints").resolve("log_reg.onnx")
            return Config(dicPath, affPath, modelPath)
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
    if (args.isEmpty()) {
        println("Pass path to test data as an argument")
        return
    }
    if (Path(args[0]).notExists()) {
        println("Passed argument is not a valid file")
        return
    }

    val config = Config.buildDefaultConfig()
    val spellChecker = SpellChecker(config.dicPath, config.affPath, config.modelPath)
    val results = File(args[0]).readLines().map { line ->
        val (misspell, correct) = line.split("\t")
        val orderedCandidates = safeRetrieveOrderedCandidates(spellChecker, misspell) ?: return@map null
        val isTop1 = orderedCandidates[0].word == correct
        val isTop5 = orderedCandidates.take(5).map { it.word }.contains(correct)
        Pair(isTop1, isTop5)
    }
    val total = results.size
    val skipped = total - results.filterNotNull().size
    val acc1 = results.filterNotNull().count { it.first }.toFloat() / total * 100
    val acc5 = results.filterNotNull().count { it.second }.toFloat() / total * 100
    println("Evaluate $total examples, skipped $skipped.\n" +
            "Accuracy@1: ${acc1.round(2)}, accuracy@5: ${acc5.round(2)}.")
}
