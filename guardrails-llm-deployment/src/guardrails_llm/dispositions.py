from enum import StrEnum


class ResponseDisposition(StrEnum):
    ANSWER = "answer"
    BLOCK = "block"
    ABSTAIN = "abstain"
    REDIRECT = "redirect"
