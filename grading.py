def gradeExam(detectedAnswers, answerKey):
    totalQuestions = len(answerKey)
    correctCount = 0
    for quesIdx, correctOption in answerKey.items():
        detectedOption = detectedAnswers.get(quesIdx, None)
        if detectedOption == correctOption:
            correctCount += 1

    print(f"Detected: {detectedAnswers}")
    print(f"Correct: {correctCount}/{totalQuestions}")
    score = (correctCount / totalQuestions) * 10
    return score