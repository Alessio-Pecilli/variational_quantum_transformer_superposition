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
    _printed_checks = False

    def __init__(
        self,
        sentences=None,
        embeddingDim=16,
        usePretrained=False,
        embeddingSeed=0,
        embeddingMatrix=None,
        rotationMatrix=None,
    ):
        self.sentences = [s.split() for s in sentences] if sentences else []
        self.embeddingDim = embeddingDim
        self.usePretrained = usePretrained
        self.embeddingSeed = embeddingSeed
        self.vocabulary = self._buildVocabulary()
        self.vocabSize = len(self.vocabulary)
        self.model = self._loadModel()
        if embeddingMatrix is None:
            embeddingMatrix = self._buildEmbeddingMatrix()
        if rotationMatrix is None:
            rotationMatrix = self._buildRotationMatrix()
        self.set_embedding_matrix(
            embeddingMatrix,
            rotation_matrix=rotationMatrix,
            isometrize=True,
        )
        if not Encoding._printed_checks:
            self._print_embedding_checks()
            Encoding._printed_checks = True

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

    def _buildRotationMatrix(self):
        rng = np.random.default_rng(self.embeddingSeed + 1)
        raw = rng.standard_normal((self.embeddingDim, self.embeddingDim))
        q, r = np.linalg.qr(raw)
        diag = np.sign(np.diag(r))
        diag[diag == 0] = 1.0
        return q * diag

    def _print_embedding_checks(self):
        e_isometric = np.allclose(
            self.embeddingMatrix.T @ self.embeddingMatrix,
            np.eye(self.embeddingDim)
        )
        f_isometric = np.allclose(
            self.outputEmbeddingMatrix.T @ self.outputEmbeddingMatrix,
            np.eye(self.embeddingDim)
        )
        range_equal = np.allclose(
            self.outputEmbeddingMatrix @ self.outputEmbeddingMatrix.T,
            self.embeddingMatrix @ self.embeddingMatrix.T
        )
        v_unitary = np.allclose(
            self.rotationMatrix.T @ self.rotationMatrix,
            np.eye(self.embeddingDim)
        )
        det_v = np.linalg.det(self.rotationMatrix)
        det_v_abs = float(np.abs(det_v))
        ef_equal = np.allclose(self.embeddingMatrix, self.outputEmbeddingMatrix)
        ef_diff = float(np.linalg.norm(self.embeddingMatrix - self.outputEmbeddingMatrix))
        print(f"[EMBEDDING] Check 1 (E^T E == I): {e_isometric}")
        print(f"[EMBEDDING] Check 2 (F^T F == I): {f_isometric}")
        print(f"[EMBEDDING] Check 3 (F F^T == E E^T): {range_equal}")
        print(f"[EMBEDDING] Check 4 (V^T V == I): {v_unitary}")
        print(f"[EMBEDDING] det(V) = {det_v:.6f} |det(V)| = {det_v_abs:.6f}")
        print(f"[EMBEDDING] E != F: {not ef_equal} |E-F|={ef_diff:.6f}")

    @staticmethod
    def isometrize_matrix(matrix):
        if matrix.shape[0] < matrix.shape[1]:
            raise ValueError("Cannot isometrize when rows < columns.")
        q, r = np.linalg.qr(matrix)
        diag = np.sign(np.diag(r))
        diag[diag == 0] = 1.0
        return q * diag

    def set_embedding_matrix(self, embedding_matrix, rotation_matrix=None, isometrize=True):
        matrix = np.asarray(embedding_matrix, dtype=np.float64)
        expected_shape = (self.vocabSize, self.embeddingDim)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"Embedding matrix shape must be {expected_shape}, got {matrix.shape}."
            )
        if rotation_matrix is None:
            if hasattr(self, "rotationMatrix"):
                rotation_matrix = self.rotationMatrix
            else:
                rotation_matrix = self._buildRotationMatrix()
        rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
        rotation_shape = (self.embeddingDim, self.embeddingDim)
        if rotation_matrix.shape != rotation_shape:
            raise ValueError(
                f"Rotation matrix shape must be {rotation_shape}, got {rotation_matrix.shape}."
            )
        if isometrize:
            matrix = self.isometrize_matrix(matrix)
            rotation_matrix = self.isometrize_matrix(rotation_matrix)
        self.embeddingMatrix = matrix
        self.embeddingMatrix.setflags(write=False)
        self.rotationMatrix = rotation_matrix
        self.rotationMatrix.setflags(write=False)
        self.outputEmbeddingMatrix = self.embeddingMatrix @ self.rotationMatrix
        self.outputEmbeddingMatrix.setflags(write=False)
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
        targets = []
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
            targets.append(self.outputEmbeddingMatrix[idx])

        for i, base in enumerate(embeddings):
            full_inputs.append(base + posEnc[i])

        return full_inputs, targets

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
