import numpy as np

class Encoding:
    def __init__(self, sentences=None, embeddingDim=16, usePretrained=False):
        self.sentences = [s.split() for s in sentences] if sentences else []
        self.embeddingDim = embeddingDim
        self.usePretrained = usePretrained
        self.vocabulary = self._buildVocabulary(self.sentences)
        self.model = self._loadModel()
        self.embeddingMatrix = self._buildEmbeddingMatrix()

    # ============================================================
    # Core: costruzione dizionario e embedding casuali
    # ============================================================
    def _buildVocabulary(self, sentences):
        vocab = {}
        idx = 0
        for sentence in sentences:
            for word in sentence:
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
        matrix = np.zeros((vocabSize, self.embeddingDim))
        for word, idx in self.vocabulary.items():
            if self.model and word in self.model:
                matrix[idx] = self.model[word][:self.embeddingDim]
            else:
                matrix[idx] = np.random.uniform(0, 0.1, self.embeddingDim)
        return matrix

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
        """Restituisce embedding + positional encoding normalizzati per UNA sola frase."""
        words = sentence.split()
        embeddings = []
        for word in words:
            if word not in self.vocabulary:
                # aggiungi dinamicamente parola nuova
                idx = len(self.vocabulary)
                self.vocabulary[word] = idx
                new_vec = np.random.uniform(0, 0.1, self.embeddingDim)
                self.embeddingMatrix = np.vstack([self.embeddingMatrix, new_vec])
            idx = self.vocabulary[word]
            embeddings.append(self.embeddingMatrix[idx])

        embeddings = np.array(embeddings)
        posEnc = self._positionalEncoding(len(words))
        combined = embeddings + posEnc

        # normalizza ogni vettore
        normalized = [v / np.linalg.norm(v) for v in combined]
        return normalized

    def localPsi(self, sentence, wordIdx):
        """Crea psi per una frase singola (stesso comportamento di prima ma per frase diretta)."""
        dim = self.embeddingDim
        phrase = self.encode_single(sentence)[:wordIdx]
        psi = np.zeros(dim * dim)
        for t in phrase:
            t = t / np.linalg.norm(t)
            psi += np.kron(t, t)
        return psi / np.linalg.norm(psi)

    def getAllPsi(self, sentence):
        """Calcola tutti i psi di una frase (equivalente a prima ma lazy)."""
        phrase = self.encode_single(sentence)
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
