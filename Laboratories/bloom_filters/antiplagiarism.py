import numpy as np
import math
import re
from pympler import asizeof
import hashlib
import matplotlib.pyplot as plt
#%%
def read_clean_data(data_path):
    """
    Apply preprocessing on the input data (i.e. remove blank spaces, punctuation, etc..)
    :param data_path: path of the inout txt file
    :return: preprocessed data
    """
    number_of_words = 0
    with open(data_path,'r') as file:
        data = file.read()

    data = re.sub(' +', ' ', data)
    data = re.sub('[\n]+', '\n', data)
    data = re.sub(r"[^\w\d'\s]+", '', data)
    data = data.lower()
    lines = data.split("\n")

    for line in lines:
        number_of_words += len(line.split())

    print("# of words:", number_of_words)
    print('# of verses:', len(lines))
    print('# of distinct verses:',len(set(data.split())))

    return data
#%%
def create_sentences(data, n):
    """
    create sentences of n-words
    :param data: input data
    :param n: the number of words in each sentence
    :return: set of n-word sentences
    """
    words = data.split()
    sentences = set()
    for i in range(len(data.split()) - n):
        sentence = " ".join(words[i:i+n])
        sentences.add(sentence)

    mem_occupancy = asizeof.asizeof(sentences)
    print("number of distinct sentences: ", len(sentences))
    print(f"actual memory occupancy of the set: {mem_occupancy} B ({mem_occupancy/(2**20)} MB)")

    return set(sentences)
#%%
text = read_clean_data("divina_commedia.txt")
sentences = create_sentences(text, 6)
avg_size = 0
for sentence in sentences:
    avg_size += asizeof.asizeof(sentence)
avg_size = avg_size / len(sentences)

print(f"Average size of each sentence: {avg_size} B")
#%%
def get_hash(sentence, n):
    """
    compute an hash function in the range [0, n)
    :param sentence: input of the hash function
    :param n: range of the hash function (n=2^b)
    :return: hash function
    """
    sentence_hash = hashlib.md5(sentence.encode('utf-8')) # md5 hash
    sentence_hash_int = int(sentence_hash.hexdigest(), 16) # md5 hash in integer format (convert to integer just to take less value since we don't need 128 bits hash
    h = sentence_hash_int % n # map into [0,n-1]
    return h
#%%
def get_bits_fingerprint(m, epsilon):
    """
    compute the number of bits needed to store m elements with a P(FP)=epsilon
    :param m: element to be stored
    :param epsilon: P(FP)
    :return: b
    """
    b = math.log(m/epsilon, 2)
    return math.ceil(b)  # round up to the next integer

def get_range_fingerprint(m, epsilon):
    """
    Compute the range [0, n) of a fingerprint needed to store m elements with a P(FP)=epsilon
    :param m: element to be stored
    :param epsilon: P(FP)
    :return: n
    """
    n = m/epsilon
    return math.ceil(n)  # round up to the next integer

def get_epsilon_fingerprint(m, b):
    """
    return the approximated value of P(FP) when storing m elements using b bits fingerprints
    :param m: element to be stored
    :param b: # bits
    :return: P(FP)
    """
    return m / 2**b
#%%
class FingerprintSet:
    """
    Class that implements a set of fingerprints
    """
    def __init__(self, data, b):
        self.data = data
        self.b = b
        self.n = 2**self.b
        self.m = len(self.data)  # maybe use len(data) instead

        self.fingerprint_set = self.fingerprints_set()

    def fingerprints_set(self):
        """
        compute the hash function for each sentence and store it in a set
        :return: set of fingerprints
        """
        fingerprints_set = set()
        # m = len(self.data)
        # n = get_range_fingerprint(m, self.epsilon)
        # print(m, n)
        for line in self.data:
            h = get_hash(line, self.n)
            fingerprints_set.add(h)

        return fingerprints_set

    def get_fingerprints_set(self):
        return self.fingerprints_set()

    def get_p_collision_th(self, p=0.5):
        """
        compute the theoretical number of bits b needed to have probability of collision p
        :param p: probability of collision
        :return: b
        """
        # 1-p = exp{-m**2 / 2n)
        # n = 2**b
        n = -(self.m**2) / (2*math.log(1-p))
        b = math.log(n, 2)

        return math.ceil(b)

    def is_collision(self):
        """
        compute if there is a collision
        """
        if len(self.data) == len(self.fingerprint_set):
            print("No collisions")
            return False
        else:
            print(f"{len(self.data) - len(self.fingerprint_set)} collisions")
            return True

    def get_collisions(self):
        """
        compute the number of collisions
        """
        return len(self.data) - len(self.fingerprint_set)

    def probability_fp(self, n):
        """
        compute the probability of false positives
        :param n: range of the hash functions [0, n)
        :return: P(FP)
        """
        return 1 - (1 - 1/n)**self.m

    def get_actual_size(self):
        """
        compute the actual memory occupancy of the fingerprints set in Bytes
        """
        return asizeof.asizeof(self.fingerprint_set)
#%%
class FingerprintSetSimulator:
    def __init__(self):
        self.text = read_clean_data("divina_commedia.txt")
        self.sentences = create_sentences(self.text, 6)
        self.m = len(self.sentences)

        self.res = dict()

    def simulate(self):
        """
        create a multiple set of fingerprints starting from b=16 until there are no collisions
        (until B_exp is found) and compute the following statistics for the optimal value of b:
        - P(FP)
        - theoretical # of bits to have no collisions
        - memory occupancy
        """
        for b in range(16, 50): # starts from 16 bit fingerprints
            fingerprint_set = FingerprintSet(self.sentences, b)
            self.res[b] = fingerprint_set.get_collisions()

            if fingerprint_set.is_collision() is False:
                break

        print()
        print("# bits: ", b)
        print("theoretical # bits: ", fingerprint_set.get_p_collision_th())
        print("P(FP): ", fingerprint_set.probability_fp(2**b))
        print(f"actual memory occupancy: {fingerprint_set.get_actual_size()} B ({fingerprint_set.get_actual_size()/(2**20)} MB)")


    def plot(self):
        plt.figure(figsize=(10,7))
        plt.plot(list(self.res.keys()), list(self.res.values()), marker="o")
        plt.grid()
        plt.xlabel("# bits", fontsize=14)
        plt.ylabel("# collisions", fontsize=14)
        plt.savefig("n_collisions_fingerprint.png")
        plt.show()
#%%
s = FingerprintSetSimulator()
s.simulate()
s.plot()
#%%
class BitStringArray:
    """
    Class that implements a bit string array
    """
    def __init__(self, data, n):
        self.data = data
        self.n = n
        self.bit_array = np.zeros(shape=self.n)
        self.m = len(data)

    def fill_array(self):
        """
        fill the bit string array computing the hash function for each sentence and setting
        to 1 the value in position h(x)
        """
        for line in self.data:
            h = get_hash(line, self.n)
            self.bit_array[h] = 1

    def probability_fp(self):
        """
        compute the P(FP)
        :return: P(FP)
        """
        return np.sum(self.bit_array) / len(self.bit_array)

    def get_actual_size(self):
        """
        compute the actual memory occupancy of the bit string array in Bytes
        """
        return asizeof.asizeof(self.bit_array)

    def probability_fp_th(self):
        """
        compute the theoretical probability of false positives
        :return: P(FP)
        """
        return 1 - (1 - 1/self.n)**self.m
#%%
class BitStringArraySimulator:
    def __init__(self):
        self.text = read_clean_data("divina_commedia.txt")
        self.sentences = create_sentences(self.text, 6)

        self.res = dict()

    def simulate(self):
        """
        create multiple bit string array for different values of b and compute the following statistics:
        - P(FP) (experimental and theoretical)
        - memory occupancy
        """
        for b in [19, 20, 21, 22, 23]:
            bit_array = BitStringArray(self.sentences, 2**b)
            bit_array.fill_array()
            p_fp = bit_array.probability_fp()
            # self.res[b] = (p_fp, get_epsilon_fingerprint(len(sentences), b))
            self.res[b] = (p_fp, bit_array.probability_fp_th())

            print("b: ", b)
            print("n: ", 2**b)
            print("P(FP): ", p_fp)
            print(f"actual memory occupancy: {bit_array.get_actual_size()} B ({bit_array.get_actual_size()/(2**20)} MB)")
            print("---------------------------")

    def plot(self):
        p_fp_exp = []
        p_fp_th = []
        keys = []
        for k, v in self.res.items():
            p_fp_exp.append(v[0])
            p_fp_th.append(v[1])
            keys.append(k)

        plt.figure(figsize=(10,7))
        plt.plot(keys, p_fp_exp, marker="o", label="simulated")
        plt.plot(keys, p_fp_th, marker="x", linestyle="--", label="theoretical")
        plt.grid()
        plt.legend()
        plt.xlabel("# bits", fontsize=14)
        plt.ylabel("P(false positives)", fontsize=14)
        plt.savefig("p_fp_bitStringArray.png")
        plt.show()
#%%
b = BitStringArraySimulator()
b.simulate()
b.plot()
#%%
class BloomFilter:
    """
    Class that implements a bloom filter
    """
    def __init__(self, data, b, k):
        self.data = data
        self.b = b
        self.n = 2**self.b
        self.m = len(self.data)
        self.k = k

        self.bloom_filter = np.zeros(shape=self.n)
        self.estimated_stored_elements = self.fill_filter()  # fill the filter

    def insert(self, x):
        """
        insert element in a bloom filter
        :param x: the element to be inserted
        """
        for i in range(self.k):
            if i == 0:
                h = get_hash(x, self.n)
            else:
                new_x = x + str(i)
                h = get_hash(new_x, self.n)

            self.bloom_filter[h] = 1

    def fill_filter(self):
        """
        fill the bloom filter with all the sentences
        """
        estimated_stored_elements = []

        for line in self.data:
            self.insert(line)
            estimated_stored_elements.append(self.distinct_elements_stored())

        return estimated_stored_elements

    def search(self, x):
        """
        search an element in the bloom filter
        :param x: the element to search
        :return: if x is found or not
        """
        ans = True
        for i in range(self.k):
            if i == 0:
                h = get_hash(x, self.n)
            else:
                x = x + str(i)
                h = get_hash(x, self.n)

            if self.bloom_filter[h] == 0:
                ans = False
                break

        return ans

    def distinct_elements_stored(self):
        """
        compute an estimation of the stored elements in the bloom filter
        """
        N = np.sum(self.bloom_filter)
        return -self.n/self.k * math.log(1 - N/self.n)


    def probability_fp(self):
        """
        compute the P(FP)
        :return: P(FP)
        """
        return (np.sum(self.bloom_filter) / len(self.bloom_filter))**self.k

    def get_estimated_stored_elements(self):
        return self.estimated_stored_elements

    def get_actual_size(self):
        """
        compute the actual memory occupancy of the bit string array in Bytes
        """
        return asizeof.asizeof(self.bloom_filter)
#%%
class BloomFilterSimulation:
    def __init__(self):
        self.text = read_clean_data("divina_commedia.txt")
        self.sentences = create_sentences(self.text, 6)

        self.res = dict()
        self.res_stored_elements = dict()

    def optimal_number_hash_functions(self, b):
        """
        compute the optimal number of hash function for a bloom filter using b bits hash functions
        (since to value are computed, we select the best one choosing the one with a lower
            theoretical probability of false positives)
        :param b: # bits of the hash functions
        :return: k, P(FP)
        """
        n = 2**b
        m = len(self.sentences)

        k = n/m * math.log(2)
        # two possible values as best
        k1 = math.floor(k)
        k2 = math.ceil(k)
        # compute the theoretical P(FP) of both values and use
        # the one with a lower probability
        p_fp1 = self.probability_fp_th(k1, m, n)
        p_fp2 = self.probability_fp_th(k2, m, n)

        if p_fp1 > p_fp2:
            return k2, p_fp2
        else:
            return k1, p_fp1

    def compute_optimal_number_has_functions(self):
        """
        compute the optimal number of hash functions for different values of b
        :return: dictionary with the best k for each value of b,
                 dictionary with the best P(FP) for each value of b
        """
        res_k = dict()
        res_p = dict()

        for b in [19, 20, 21, 22, 23]:
            k, p_fp = self.optimal_number_hash_functions(b)
            res_k[b] = k
            res_p[b] = p_fp
            print("b: ", b)
            print("\tp_fp: ", p_fp)
            print("\toptimal k: ", k)

        return res_k, res_p

    def probability_fp_th(self, k, m, n):
        """
        compute the theoretical P(FP) of a bloom filter
        :param k: # hash functions
        :param m: # of elements to store
        :param n: # range of the hash functions [0, n), n=2^b
        :return: P(FP)
        """
        return (1 - (1 - 1/n)**(m*k))**k

    def plot_optimal_k(self):
        res_k, res_p = self.compute_optimal_number_has_functions()

        plt.figure(figsize=(10,7))
        plt.plot(list(res_k.keys()), list(res_k.values()), marker="o")
        plt.grid()
        plt.xlabel("# bits", fontsize=14)
        plt.ylabel("k (optimal number of hash functions)", fontsize=14)
        plt.savefig("optimal_k.png")
        plt.show()

    def plot_optimal_k_p(self):
        res_k, res_p = self.compute_optimal_number_has_functions()

        fig,ax = plt.subplots(figsize=(10,7))
        ax.plot(list(res_k.keys()), list(res_k.values()), color='#1f77b4', marker="o")
        ax.set_xlabel("# bits", fontsize = 14)
        ax.set_ylabel("k (optimal number of hash functions)", color='#1f77b4', fontsize=14)
        ax2=ax.twinx()
        ax2.plot(list(res_p.keys()), list(res_p.values()),color='#ff7f0e', linestyle=':', marker="x")
        ax2.set_ylabel("theoretical P(false positive)",color='#ff7f0e', fontsize=14)
        ax.grid()
        plt.savefig("optimal_k.png")
        plt.show()

    def simulate(self):
        """
        create multiple bloom filter for different values of b and compute the following statistics:
        - P(FP) (experimental and theoretical)
        - estimated number of stored elements
        - memory occupancy
        """
        for b in [19, 20, 21, 22, 23]:
            k, th_p = self.optimal_number_hash_functions(b)  # compute optimal k
            bloom_filter = BloomFilter(self.sentences, b, k)  # create bloom filter

            self.res[b] = (bloom_filter.probability_fp(), th_p)  # evaluate the probability of false positive
            self.res_stored_elements[b] = bloom_filter.get_estimated_stored_elements()

            print("b: ", b)
            print("P(FP): ", bloom_filter.probability_fp())
            print(f"actual memory occupancy: {bloom_filter.get_actual_size()} B ({bloom_filter.get_actual_size()/(2**20)} MB)")
            print("---------------------------")

    def plot(self):
        p, th_p = [], []
        for _p, _th_p in self.res.values():
            p.append(_p)
            th_p.append(_th_p)

        plt.figure(figsize=(10,7))
        plt.plot(list(self.res.keys()), p, marker="o", label='simulated')
        plt.plot(list(self.res.keys()), th_p, marker="x", linestyle=':', label='theoretical')
        plt.grid()
        plt.xlabel("# bits", fontsize=14)
        plt.ylabel("P(false positives)", fontsize=14)
        plt.legend()
        plt.savefig("p_fp_bloom_filters.png")
        plt.show()

    def plot_inserted_elements_error(self):
        plt.figure(figsize=(10,7))

        for k, v in self.res_stored_elements.items():
            plt.plot(v - np.linspace(1, len(v), len(v)), label=f"b={k}")

        plt.grid()
        plt.legend()
        plt.xlabel("number of inserted element", fontsize=14)
        plt.ylabel("estimation of inserted elements error", fontsize=14)
        plt.savefig("estimation_inserted_elements_error.png")
        plt.show()

#%%
bf = BloomFilterSimulation()
#%%
bf.plot_optimal_k_p()
#%%
bf.simulate()
bf.plot()
#%%
bf.plot_inserted_elements_error()
#%%
set_of_sentences_mem = 12987584/2**10

fingerprints_mem = {32: 7288920/2**10}
fingerprints_p_fp = {32: 2.2514469789514102e-05}

bitstring_array_mem = {
    19: 4194432/2**10,
    20: 8388736/2**10,
    21: 16777344/2**10,
    22: 33554560/2**10,
    23: 67108992/2**10
}
bitstring_array_p_fp = {
    19: 0.16833877563476562,
    20: 0.0881052017211914,
    21: 0.04510641098022461,
    22: 0.02279829978942871,
    23: 0.011467814445495605
}

bloom_filter_mem = {
    19: 4194432/2**10,
    20: 8388736/2**10,
    21: 16777344/2**10,
    22: 33554560/2**10,
    23: 67108992/2**10
}
bloom_filter_p_fp = {
    19: 0.07436877992190073,
    20: 0.005492450660940125,
    21: 3.0026682166714118e-05,
    22: 8.916870695240416e-10,
    23: 7.8527901157876865e-19
}
#%%
plt.figure(figsize=(10,7))
#plt.axhline(y=set_of_sentences_mem, label="set of sentences")
plt.scatter(0, set_of_sentences_mem, s=100)
plt.scatter(list(fingerprints_p_fp.values()), list(fingerprints_mem.values()), marker="x", label="fingerprints", s=100, linewidths=3)
plt.axhline(y=set_of_sentences_mem, linestyle="--", label="set of sentences")
# plt.axhline(y=list(fingerprints_mem.values())[0], linestyle="--", color='#ff7f0e', label="fingerprints")
plt.plot(list(bitstring_array_p_fp.values()), list(bitstring_array_mem.values()), label="bit-string array", marker="x", color='#2ca02c', markersize=10)
plt.plot(list(bloom_filter_p_fp.values()), list(bloom_filter_mem.values()), label="bloom filter", marker="x", color='#d62728', markersize=10)

for k in bitstring_array_p_fp.keys():
    plt.annotate(f"  b={k}", (bitstring_array_p_fp[k], bitstring_array_mem[k]), fontsize=10)
    plt.annotate(f"  b={k}", (bloom_filter_p_fp[k], bloom_filter_mem[k]), fontsize=10)
for k in fingerprints_p_fp.keys():
    # plt.annotate(f"  b={k}", (fingerprints_p_fp[k], fingerprints_mem[k]), fontsize=10)
    plt.text(fingerprints_p_fp[k]+.001, fingerprints_mem[k]-2200, f"b={k}", fontsize=10)

plt.grid()
plt.legend()
plt.xlabel("P(false positives)", fontsize=14)
plt.ylabel("Actual memory occupancy [kB]", fontsize=14)
plt.savefig("summary.png")
plt.show()