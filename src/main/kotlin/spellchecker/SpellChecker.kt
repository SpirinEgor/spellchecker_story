package spellchecker

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import dumonts.hunspell.Hunspell
import info.debatty.java.stringsimilarity.Damerau
import info.debatty.java.stringsimilarity.JaroWinkler
import info.debatty.java.stringsimilarity.LongestCommonSubsequence
import java.nio.file.Path
import kotlin.math.max

class SpellChecker(dictionaryPath: Path, affixPath: Path, onnxModelPath: Path) {

    private val hunspell = Hunspell(dictionaryPath, affixPath)
    private val onnxEnvironment = OrtEnvironment.getEnvironment()
    private val onnxSession = onnxEnvironment.createSession(onnxModelPath.toString(), OrtSession.SessionOptions())
    private val damerauLevenshtein = Damerau()
    private val jaroWinkler = JaroWinkler()
    private val longestCommonSubsequence = LongestCommonSubsequence()

    private fun buildCandidateFeatures(candidate: String, word: String): FloatArray {
        val maxLength = max(candidate.length, word.length)
        val levenshteinDist = (1 - damerauLevenshtein.distance(word, candidate)).toFloat() / maxLength
        val jaroWinklerDist = (1 - jaroWinkler.similarity(word, candidate)).toFloat() / maxLength
        val lcsDist = longestCommonSubsequence.length(word, candidate).toFloat() / maxLength
        return floatArrayOf(levenshteinDist, jaroWinklerDist, lcsDist)
    }

    private fun extractProbabilities(features: Array<FloatArray>): List<Float> {
        val onnxInputTensor = OnnxTensor.createTensor(onnxEnvironment, features)
        val onnxInputs = mapOf("input" to onnxInputTensor)
        val results = onnxSession.run(onnxInputs);

        val labels = results.get(1).value as ArrayList<*>
        // FIXME: getter with key 1 return null ¯\_(ツ)_/¯
        return labels.map { (it as HashMap<Int, Float>).values.toList()[1] }.toList()
    }

    fun correct(word: String): List<Candidate> {
        val candidates = hunspell.suggest(word)
        if (candidates.isEmpty())
            return listOf(Candidate(word, 0.0F))
        val features = candidates.map { buildCandidateFeatures(it, word) }.toTypedArray()
        val probabilities = extractProbabilities(features)
        return candidates.zip(probabilities).map { Candidate(it.first, it.second) }
    }
}
