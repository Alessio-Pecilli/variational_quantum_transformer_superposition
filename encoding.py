import numpy as np

from config import (
    SUBJECT_ADJS,
    SUBJECTS,
    ADVERBS,
    VERBS,
    DETERMINERS2,
    PLACE_ADJS,
    PLACES,
    ENDINGS,
)

class Encoding:
    def __init__(self, sentences=None, embeddingDim=16, usePretrained=False, embeddingSeed=0, embeddingMatrix=None):
        self.sentences = [s.split() for s in sentences] if sentences else []
        self.embeddingDim = embeddingDim
        self.usePretrained = usePretrained
        self.embeddingSeed = embeddingSeed
        self.vocabulary = self._buildVocabulary()
        self.vocabSize = len(self.vocabulary)
        self.model = self._loadModel()
        if embeddingMatrix is None:
            embeddingMatrix = self._buildEmbeddingMatrix()
            self.set_embedding_matrix(embeddingMatrix, isometrize=False)
        else:
            self.set_embedding_matrix(embeddingMatrix, isometrize=True)
        is_isometric = np.allclose(
            self.embeddingMatrix.T @ self.embeddingMatrix, np.eye(self.embeddingDim)
        )
        print(is_isometric)

    # ============================================================
    # Core: costruzione dizionario e embedding casuali
    # ============================================================
    def _buildVocabulary(self):
        vocab = {}
        idx = 0
        vocab_sources = [
            ["<UNK>", "The"],
            SUBJECT_ADJS,
            SUBJECTS,
            ADVERBS,
            VERBS,
            DETERMINERS2,
            PLACE_ADJS,
            PLACES,
            ENDINGS,
        ]
        for words in vocab_sources:
            for word in words:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab

    def _loadModel(self):
        if self.usePretrained:
            print("Warning: Pretrained models disabled. Using random embeddings.")
        return None

    def _buildEmbeddingMatrix(self):
        vocabSize = len(self.vocabulary)
        if vocabSize < self.embeddingDim:
            raise ValueError(
                "Vocabulary size must be >= embeddingDim to build an isometric matrix."
            )
        rng = np.random.default_rng(self.embeddingSeed)
        embedding, _ = np.linalg.qr(rng.standard_normal((vocabSize, self.embeddingDim)))
        return embedding

    @staticmethod
    def isometrize_matrix(matrix):
        if matrix.shape[0] < matrix.shape[1]:
            raise ValueError("Cannot isometrize when rows < columns.")
        q, r = np.linalg.qr(matrix)
        diag = np.sign(np.diag(r))
        diag[diag == 0] = 1.0
        return q * diag

    def set_embedding_matrix(self, embedding_matrix, isometrize=True):
        matrix = np.asarray(embedding_matrix, dtype=np.float64)
        expected_shape = (self.vocabSize, self.embeddingDim)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Embedding matrix shape must be {expected_shape}, got {matrix.shape}."
            )
        if isometrize:
            matrix = self.isometrize_matrix(matrix)
        self.embeddingMatrix = matrix
        self.embeddingMatrix.setflags(write=False)
        return self.embeddingMatrix

    # ============================================================
    # Funzioni di codifica singola (no array globali)
    # ============================================================
    def _positionalEncoding(self, seqLen):
        dModel = self.embeddingDim
        position = np.arange(seqLen)[:, np.newaxis]
        divTerm = np.exp(np.arange(0, dModel, 2) * -(np.log(10000.0) / dModel))
        pe = np.zeros((seqLen, dModel))
        pe[:, 0::2] = np.sin(position * divTerm)
        pe[:, 1::2] = np.cos(position * divTerm)
        return pe

    def encode_single(self, sentence):
        """Restituisce (x, x_target) per UNA sola frase."""
        words = sentence.split()
        embeddings = []
        full_inputs = []
        unk_idx = self.vocabulary.get("<UNK>")
        posEnc = self._positionalEncoding(len(words))
        for word in words:
            if word not in self.vocabulary:
                if unk_idx is None:
                    raise KeyError(
                        f"Word '{word}' not in fixed vocabulary; embedding matrix is immutable."
                    )
                idx = unk_idx
            else:
                idx = self.vocabulary[word]
            embeddings.append(self.embeddingMatrix[idx])

        for i, base in enumerate(embeddings):
            full_inputs.append(base + posEnc[i])

        return full_inputs, embeddings

    def localPsi(self, sentence, wordIdx):
        """Crea psi per una frase singola (stesso comportamento di prima ma per frase diretta)."""
        dim = self.embeddingDim
        phrase, _ = self.encode_single(sentence)
        phrase = phrase[:wordIdx]
        psi = np.zeros(dim * dim)
        for t in phrase:
            t = t / np.linalg.norm(t)
            psi += np.kron(t, t)
        return psi / np.linalg.norm(psi)

    def getAllPsi(self, sentence):
        """Calcola tutti i psi di una frase (equivalente a prima ma lazy)."""
        phrase, _ = self.encode_single(sentence)
        dim = self.embeddingDim
        psiList = []
        for wordIdx in range(0, len(phrase) - 1):
            psi = np.zeros(dim * dim)
            for t in phrase[:wordIdx + 1]:
                t = t / np.linalg.norm(t)
                psi += np.kron(t, t)
            psi /= np.linalg.norm(psi)
            psiList.append(psi)
        return psiList
