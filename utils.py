import string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import re


# string.printable[:95]
# '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
class VectorizeChar:
    def __init__(self, max_len=100):
        syntactic_tokens = [r"_", r"^", r'{', r'}', r'&', r'\\', ' ']
        print(syntactic_tokens)
        latin_letters = [chr(i) for i in range(ord('a'), ord('z') + 1)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        print(latin_letters)
        numbers = [str(i) for i in range(10)]
        print(numbers)
        blackboard = [r'\mathbb']
        print(blackboard)
        latin_punctuations_symbols = [',', ';', ':', '!', '?', '.', '(', ')', '[', ']', r'\{', r'\}', '*', '/', '+', '-', r'\_', r'\&', r'\#', r'\%', '|', r'\backslash']
        print(latin_punctuations_symbols)
        greek_letters = r'\alpha \beta \delta \Delta \epsilon \eta \chi \gamma \Gamma \iota \kappa \lambda \Lambda \nu \mu \omega \Omega \phi \Phi \pi \Pi \psi \Psi \rho \sigma \Sigma \tau \theta \Theta \upsilon \Upsilon \varphi \varpi \varsigma \vartheta \xi \Xi \zeta'
        greek_letters = greek_letters.split(' ')
        print(greek_letters)
        math_constructs = r'\frac \sqrt \prod \sum \iint \int \oint'.split(' ')
        print(math_constructs)
        modifiers = r'\hat \tilde \vec \overline \underline \prime \dot \not'.split(' ')
        print(modifiers)
        delimiters = r'\langle \rangle \lceil \rceil \lfloor \rfloor \|'.split(' ')
        print(delimiters)
        conparisons = [r'\ge', r'\gg', r'\le', r'\ll', '<', '>']
        print(conparisons)
        eq_aprox = r'= \approx \cong \equiv \ne \propto \sim \simeq'.split(' ')
        print(eq_aprox)
        set_theory = r'\in \ni \notin \sqsubseteq \subset \subseteq \subsetneq \supset \supseteq \emptyset'.split(' ')
        print(set_theory)
        operators = r'\times \bigcap \bigcirc \bigcup \bigoplus \bigvee \bigwedge \cap \cup \div \mp \odot \ominus \oplus \otimes \pm \vee \wedge'.split(' ')
        print(operators)
        arrows = r'\hookrightarrow \leftarrow \leftrightarrow \Leftrightarrow \longrightarrow \mapsto \rightarrow \Rightarrow \rightleftharpoons \iff'.split(' ')
        print(arrows)
        dots = r'\bullet \cdot \circ'.split(' ')
        print(dots)
        others = r'\aleph \angle \dagger \exists \forall \hbar \infty \models \nabla \neg \partial \perp \top \triangle \triangleleft \triangleq \vdash \Vdash \vdots'.split(' ')
        print(others)
        print('\n\n')

        total = syntactic_tokens + latin_letters + numbers + blackboard + latin_punctuations_symbols + greek_letters + math_constructs + modifiers + delimiters + conparisons + eq_aprox + set_theory + operators + arrows + dots + others
        print(len(total))
        print(total)
        self.vocab = total
        self.max_len = len(total)
        self.char_to_idx = {}
        self._COMMAND_RE = re.compile(r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)')
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def tokenize_expression(self, s: str) -> list[str]:
        r"""Transform a Latex math string into a list of tokens.

        Tokens are strings that are meaningful in the context of Latex
        e.g. '1', r'\alpha', r'\frac'.

        Args:
            s: unicode input string (ex: r"\frac{1}{2}")

        Returns:
            tokens: list of tokens as unicode strings.
        """
        tokens = []
        while s:
            if s[0] == '\\':
                tokens.append(self._COMMAND_RE.match(s).group(0))
            else:
                tokens.append(s[0])

            s = s[len(tokens[-1]) :]

        return tokens

    def __call__(self, text):
        # text = text[: self.max_len - 2]
        # text = "<" + text + ">"
        text = self.tokenize_expression(text)
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] #+ [0] * pad_len

    def get_vocabulary(self):
        return self.vocab

class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
        self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28
    ):
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        # if epoch % 1 != 0:
        #     return
        source = self.batch["source"]
        target = self.batch["target"].numpy()
        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-','')}")
            print(f"prediction: {prediction}\n")

class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """ linear warm up - linear decay """
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * tf.cast(epoch, tf.float32)
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (tf.cast(epoch, tf.float32) - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / (self.decay_epochs),
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        return self.calculate_lr(epoch)



def get_data(df: pd.DataFrame)->list:
    """ returns mapping of handwriting paths and transcription texts """
    data = []
    for  idx,row in df.iterrows():
      data.append({"audio": row['filename'], "text": row['transcript'].strip()})
    return data

def path_to_features(path):
  """Convert a path to a array of features."""
  x = tf.io.read_file(path)
  x = tf.io.decode_raw(x, out_type = tf.float32)
  x = tf.reshape(x, [-1, 20])    
  # x = x[::2]
  x=(x-tf.keras.backend.mean( x, axis=1, keepdims=True))/tf.keras.backend.std( x, axis=1, keepdims=True)
  return x
def txt_to_labels(txt:str, vectorizer: VectorizeChar):
  """Convert a text to a array of labels."""
  return tf.convert_to_tensor(vectorizer(txt),dtype=tf.int64)






#  for evaluation

def levenshtein(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def wer(ground_truth, prediction):    
    word_distance = levenshtein(ground_truth.split(), prediction.split())
    word_length = len(ground_truth.split())
    wer = word_distance/word_length
    wer = min(wer, 1.0)
    return wer
    
def cer(ground_truth, prediction):
    char_distance = levenshtein(ground_truth, prediction)
    char_length = len(ground_truth)
    cer=char_distance/char_length
    cer = min(cer, 1.0)
    return  cer
