plugins {
    kotlin("jvm") version "1.5.31"
    application
}

repositories {
    mavenCentral()
}

dependencies {
    implementation(platform("org.jetbrains.kotlin:kotlin-bom"))
    implementation("org.jetbrains.kotlin:kotlin-stdlib-jdk8")
    // HunSpell implementation: https://gitlab.com/dumonts/hunspell-java/
    implementation("com.gitlab.dumonts:hunspell:1.1.1")
    // ONNX runtime
    implementation("com.microsoft.onnxruntime:onnxruntime:1.9.0")
    // String similarities
    implementation("info.debatty:java-string-similarity:2.0.0")

    testImplementation("org.jetbrains.kotlin:kotlin-test")
    testImplementation("org.jetbrains.kotlin:kotlin-test-junit")
}

application {
    mainClass.set("spellchecker.AppKt")
}
