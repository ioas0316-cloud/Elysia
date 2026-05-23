import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

class FractalRotorMath:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Using a lightweight sentence transformer to embed the user's thoughts
        self.encoder = SentenceTransformer(model_name)
        # We need a PCA to reduce embeddings to 3D.
        # Using IncrementalPCA or just a fixed random projection might be better
        # To avoid coordinate frame mismatch bugs when PCA basis changes, we will
        # use a fixed projection matrix (randomized, but fixed seed for consistency)
        np.random.seed(42)
        # embedding dimension of all-MiniLM-L6-v2 is 384
        self.projection_matrix = np.random.randn(384, 3) / np.sqrt(384)

    def get_text_density(self, text):
        """
        Calculate the 'density' or scalar aspect (w) of the text.
        We'll use a combination of length and unique word ratio as a proxy.
        """
        words = text.split()
        if not words:
            return 0.5
        unique_words = set(words)
        density = len(unique_words) / len(words)
        # Normalize somewhat to keep it within [0, 1]
        length_factor = min(len(words) / 50.0, 1.0)

        w = (density + length_factor) / 2.0
        return w

    def text_to_quaternion(self, text):
        """
        Maps a text input to a 4D Quaternion (w, x, y, z).
        """
        w = self.get_text_density(text)

        emb = self.encoder.encode([text])[0]

        # Fixed projection to 3D to ensure consistent coordinate frame
        xyz = np.dot(emb, self.projection_matrix)

        # Create quaternion
        q = np.array([w, xyz[0], xyz[1], xyz[2]])

        # Normalize to make it a unit quaternion (pure rotation)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        else:
            q = np.array([1.0, 0.0, 0.0, 0.0])

        return q

    @staticmethod
    def quaternion_multiply(q1, q2):
        """
        Hamilton product of two quaternions.
        q = w + xi + yj + zk
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    @staticmethod
    def quaternion_conjugate(q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    @staticmethod
    def quaternion_angle(q1, q2):
        """
        Calculate the angle theta between two unit quaternions.
        theta = 2 * acos(|q1 . q2|)
        """
        dot_product = np.abs(np.dot(q1, q2))
        # Ensure domain is within [-1, 1] for arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        theta = 2 * np.arccos(dot_product)
        return theta

    @staticmethod
    def get_direction_vector(q1, q2):
        """
        Returns the 3D direction vector representing the axis of rotation
        from q1 to q2.
        q_diff = q2 * q1_conjugate
        """
        q1_conj = FractalRotorMath.quaternion_conjugate(q1)
        q_diff = FractalRotorMath.quaternion_multiply(q2, q1_conj)

        # The axis of rotation is the vector part (x, y, z)
        v = q_diff[1:4]
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        return v
