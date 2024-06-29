import os
from collections import deque


class QAMemory:
    def __init__(self, depth):
        self.depth = depth
        self.history = deque()

    def add(self, question, answer):
        self.history.append({"Human": f"{question}", "AI": f"{answer}"})

        numOfLines = len(self.history)
        if numOfLines > self.depth:
            self.history.popleft()

    def clear(self):
        self.history.clear()

    def getHistory(self):
        newLine = os.linesep

        historicalMsg = ""
        for line in self.history:
            historicalMsg += line["Human"] + newLine
            historicalMsg += line["AI"] + newLine

        return historicalMsg
