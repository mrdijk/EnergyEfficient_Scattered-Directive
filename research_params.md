# From Centralized to FL: exploring performance and resource consumption

Symbol  and Definition

- 𝑑𝑠𝑎𝑚𝑝𝑙𝑒 Training dataset samples
- 𝑑 Training dataset size (bytes)
- 𝑧 Training dataset partitions (i.e., total clients)
- 𝑟_𝑠𝑎𝑚𝑝𝑙𝑒 Training dataset size to samples ratio
- 𝑚 ML model size (bytes)
- 𝑟_𝑑𝑎𝑡𝑎 Client data to model size ratio
- 𝑘 Per round participating clients
- 𝑞 Online clients
- 𝑡_𝑢𝑝𝑙𝑜𝑎𝑑 Total time for client data upload in each round (sec)
- 𝑡_𝑒𝑛𝑑 Maximum duration of ML task (sec)
- 𝑠_𝑐𝑙𝑖𝑒𝑛𝑡 Client throughput — access network (bytes/sec)
- 𝑐^𝐶𝐻 Average area throughput — access network (bytes/sec)
- 𝜎 Client throughput standard deviation — access network
- c^𝐶𝐻_𝑚𝑖𝑛 Minimum client throughput — access network (bytes/sec)
- 𝑠_𝑐𝑜𝑟𝑒 Core network element’s throughput (bytes/sec)
- 𝜎_𝑒𝑑 Data skewness parameter
- 𝜎_𝑖𝑖𝑑 Independently and identically distributed (i.i.d) level shape parameter
- 𝑣^𝑀𝐿_𝑐𝑙𝑖𝑒𝑛𝑡 Client device computational capacity for training (samples/sec)
- 𝑣^𝑀𝐿_𝑐𝑙𝑜𝑢𝑑 Cloud computational capacity for training (samples/sec)
- 𝑣^𝐴𝐺_𝑐𝑙𝑜𝑢𝑑 Cloud computational capacity for model aggregation (models/sec)
- 𝑒_𝑐𝑙𝑖𝑒𝑛𝑡 Total energy expenditure for all clients (J)
- 𝑝^𝑇𝑋_𝑐𝑙𝑖𝑒𝑛𝑡_𝑖 Client device power consumption for transmission (W)
- 𝑝^𝑅𝑋_𝑐𝑙𝑖𝑒𝑛𝑡_𝑖 Client device power consumption for reception (W)
- 𝑝^𝑀𝐿_𝑐𝑙𝑖𝑒𝑛𝑡_𝑖 Client device power consumption for training (W)
- 𝑒_𝑐𝑜𝑟𝑒 Total energy expenditure in the core network (J)
- 𝑒_𝑐𝑙𝑜𝑢𝑑 Total energy expenditure in the cloud (J)
- 𝑝^𝑀𝐿_𝑐𝑙𝑜𝑢𝑑 Cloud power consumption for training (W)
- 𝑝^𝐴𝐺_𝑐𝑙𝑜𝑢𝑑 Cloud power consumption for model aggregation (W)
- ℎ_𝑒𝑝𝑜𝑐ℎ𝑠 Number of epochs (ML hyperparameter)
- ℎ_𝑏𝑎𝑡𝑐ℎ Batch size (ML hyperparameter)
- ℎ_𝑟𝑎𝑡𝑒 Learning rate (ML hyperparameter)

> For the first set of experiments, they assume data is symmetric across all clients (i.e., i.i.d. and e.d. setting) and vary the number of dataset partitions i.e.,total clients 𝑧 = {15, 30, 60, 90, 120, 150, 190, 230, 260, 300, 330, 360, 400}. Therefore the data to model ratio becomes 𝑟_𝑑𝑎𝑡𝑎_𝑖 = 𝑑∕(𝑚 ⋅ 𝑧), ∀𝑖 ∈ [1, 𝑧] (see Section 3.3). Also, for the first set of experiments, they assumed a constant value for the per round participants 𝑘 = 5.

> We also explore this data is distributed across clients, *i.e.*, data heterogeneity. We focus on two dimension; variations in size are modeled by the evenly distributed level (e.d.), while variations in content are captured by the indenependent and identically distributed level (i.i.d). Generally, client data can experience various levels of e.d., i.i.d. or combinations of both.

> Image-classification on the Street View House Numbers (SVHN) dataset. This is based on real-world images, with digits taken from house numbers on Google Street View, containing 531,131 32x32 color training images split over 10 classes (digits 0 through 9) totaling 1.3 GB in size and 26,032 test images (63 MB). The linear neural net comprises of an input layer with 3072 neurons corresponding to the 32x32x3 pixels in the input images, an output layer of 10 neurons and a hidden intermediate layer of 512 neurons. A ReLu function is applied to the hidden linear layer and on the output layer a LogSoftMax function. Training loss = negative log-likelihood. Total size of the model is 6.1 MB.

> **Hyperparameter Tuning** epochs tested: {1,5,10,25,50,100,200}, max accuracy achieved with: [25,100]. Increasing epochs causes a linear increase in energy consumption. Batch sizes tested: {64,128,500,1000}, learning rates tested: {0.0001,0.0005,0.001,0.005,0.01,0.05,0.1}. Max (FL) accuracy achieved with batch <=128, rate>=0.05


> **On the e.d. level**: describes the size distribution across clients. The size of each client's dataset is modeled as a random variable (F^~) that follows Zipf's law. Represented using a Zeta distribution with density function, px = x^{\sigma_{ed}}/?^{\sigma_{ed}}. ? respresents the Riemann Zeto function, while the Zeto distribution's skewed parameter \sigma_{ed} \in (1,+\inf) shapes the e.d. level, from uniform towards high assymetry.

> **On the i.i.d. level**: If a setting with independent and identically distributed (i.i.d) data is assumed, then the data samples in each client have the same probability distribution and are mutually independent. In our image-classification problem (see Section 3.6) that would essentially mean that each user holds samples from all classes (unbiased setting). That is represented by the i.i.d. level 𝜎𝑖𝑖𝑑 , being equal to the total number of dataset classes. For some real-world scenarios, noni.i.d. (biased) settings could occur, since each participating client might not be expected to possess a representative subset of all classes in the total training dataset. To study different levels of bias, we restrict the number of dataset classes a client can hold i.e., the value of 𝜎𝑖𝑖𝑑 is smaller compared to the total number of classes. 

Kappa_list=[10]
eed_list=[1.7,2,2.3,1000]
iid_list=[7,5,3,-1]
waiting_factor_list=[0]

batch_list =[128]
local_epoch_list=[25]
learn_rate_list=[0.1]
tbegin_list=[0*3600]
agg_list=[1]

_MODEL = 'SVHN'
    
if self._MODEL == 'SVHN':
            self.central_model = SVHN_Model()
            if self._IID == -1:
                self.learning = SVHN_big(self._SUBDATASETS,self._BATCH_SIZE,self._LEARNING_RATE,self._EED)
            if self._IID == 0:
                self.learning = SVHN_big(self._SUBDATASETS,self._BATCH_SIZE,self._LEARNING_RATE,self._EED)
            elif self._IID == 3:
                self.learning = SVHN_big_noniid(self._SUBDATASETS,self._BATCH_SIZE,self._LEARNING_RATE,splits=3,sd=self._EED)
            elif self._IID == 5:
                self.learning = SVHN_big_noniid(self._SUBDATASETS,self._BATCH_SIZE,self._LEARNING_RATE,splits=5,sd=self._EED)
            elif self._IID == 7:
                self.learning = SVHN_big_noniid(self._SUBDATASETS,self._BATCH_SIZE,self._LEARNING_RATE,splits=7,sd=self._EED)
# AutoFL

|   | Description                        | Discrete Values                                        |
| - | ---------------------------------- | ------------------------------------------------------ |
|   | # of CONV layers                   | Small (<10), medium (<20), large (<30), larger (>=40)  |
|   | # of FC layers                     | Small (<10), large (>=10)                              |
|   | # of RC layers                     | Small (<5), medium (<10), large (>=10)                 |
| B | Batch size                         | Small (<8), medium (<32), large (>=32)                 |
| E | # epochs                           | Small (<5), medium (<10), large (>=10)                 |
| K | # participant devices              | Small (<10), medium (<50), large (>=50)                |
|   | CPU utilization of co-running apps | None (0%), small (<25%), medium (<75%), large (<=100%) |
|   | Memory usage of co-running apps    | None (0%), small (<25%), medium (<75%), large (<=100%) |
|   | Network bandwidth                  | Regular (>40Mbps), bad (<=40Mbps)                      |
|   | # data classes for a given round   | Small (<25%), medium (<100%), large (=100%)            |

200 mobile devices (30 high-end, 70 mid-end, 100 low-end)

> Data distribution: We emulate different levels of data hetero-geneity by distributing the total training dataset in four differentways [ 15 , 75 ]: Ideal IID, Non-IID (50%), Non-IID (75%), and Non-IID(100%). In case of Ideal IID, all the data classes are evenly distributedto the devices in the cluster. On the other hand, in case of Non-IID (M%), M% of total devices have non-IID data while the rest haveIID samples of all data classes. For non-IID devices, we distributeeach data class randomly following a Dirichlet distribution with a0.1 concentration parameters [15 , 57 , 72 , 75 , 78 ] — the smaller thevalue of the concentration parameter, the more each data class isconcentrated on one device.

Cluster of devices used for characterization.Cluster H M L Policy

C0 - - - FedAvg-Random (Baseline)

C1 20 0 0 Performance

C2 15 5 0

C3 10 5 5

C4 5 10 5

C5 5 5 10

C6 0 5 15

C7 0 0 20 Power

Global parameter settings.

Setting B E K

S1 32 10 20

S2 32 5 20

S3 16 5 20

S4 16 5 10

# Toward Resource-Efficient Federated Learning in Mobile Edge Computing

> In the experiment, we consider the scenario of federated learning with 64 mobile clients for image classification tasks on the MNIST dataset. For the 70,000 samples in the MNIST dataset, 60,000 of them are used for training and the others for testing. The training data are assigned to the clients in two ways. For the IID setting, we shuffle the data and uniformly divide it to all clients, each with 937 samples. For the non-IID setting, we sort the data by labels, split them into 320 shards, with 5 shards for each client. In the experiment, the global CNN model has the structure {16C3-32C3-MP2-32C3-MP2-10C3-GAP-10SM}, with partitioning factors {2, 2, 2, 1} for the convolutional layers, respectively.1 The hyperparameters are set to batch size 10, local epoch number 1, learning rate 0.05, and decay rate per round 0.996.

> We compare our module-based federated learning (MFL) with the traditional federated learning (TFL) in test accuracy, energy consumption, and convergence time. Two power conditions, low power (LP) mode with power budget Pbud = 10 W and high power (HP) mode with power budget Pbud = 100 W, are considered, and accordingly, the optimal model partitions {16L, 16M, 16S, 16T}, {32L, 16M, 8S, 8T} are searched for them, respectively. Here, #L, M, S, T, denote the number of submodels in “Large,” “Medium,” “Small,” and “Tiny” sizes that have convolutional layers {16C3, 32C3, 32C3, 10C3}, {16C3, 16C3, 32C3, 10C3}, {16C3, 16C3, 16C3, 10C3}, {8C3, 16C3, 16C3, 10C3}, respectively.

# Toward Energy-Efficient Distributed Federated Learning for 6G Networks

- Scenario #1 CVFL
- Scenario #2 DBFL-Homogeneous
- Scenario #3 DBFL-Heterogeneous

> We consider three scenarios, as shown in Fig.3. All the scenarios consider one aerial base station and five devices in the coverage area. The number of devices was set to fi ve for dealing with a relatively large but manageable and easy-to-understand number of devices, and for evenly dividing them into two clusters considering the example shown in Fig. 2.

---

> The first scenario represents the CVFL approach  where each of the devices in the area needs to connect to the aerial base station. However, it can be seen that two out of five devices in this scenario cannot connect to the aerial base station due to high  latency issues.

> The second scenario represents the DBFL approach where devices can connect with a device head in a cluster, which then sends the aggregated model to the aerial base station. This  scenario uses a homogeneous feature space, and for the sake of simplicity, we show diff erent hypothetical D2D clusters without performing actual operations.

> The third scenario represents the DBFL  approach with similar characteristics as the second scenario, but a heterogeneous feature space is used. We employ the IoT device type identification dataset proposed in [15] that has nine diff erent types of IoT devices. The reason for the selection of the said dataset is twofold. First is the availability of the dataset in the public domain, and second is the compliance of dataset characteristics with the domain of  the proposed study (i.e., futuristic networks).

---

> For homogeneous CVFL and DBFL, it is assumed that each device is trained with an artificial neural network having 80 neurons and 3500 data samples. The maximum transmission time is set to 0.1s;  beyond this time, the device cannot connect to either the aerial base station or the device head.

> For heterogeneous DBFL, we assume that each device uses different dimensions of features but undergoes AE to get the unified feature space dimension. For instance, the employed dataset has around 274 features. One device may use 60 features, while the others might use numbers of features within the range of 1–274. These features will be the input to the AEs, which generates, say, 15 dimensions of feature space. For the sake of simplicity, we chose the output of 25, while the rest of the parameters are the same as Scenario #2.

# Cost-Effective Federated Learning Design

> As illustrated in Fig. 1, we consider the number of participating clients (K) and the number of local iterations (E) in each FL round as our control variables. A similar methodology can be applied to analyze problems with other control variables as well. We analyze, for the first time, how to design adaptive FL that optimally chooses K and E to minimize the total cost while ensuring convergence.
>
> Similar to existing works [2], [10], [11], we sample K clients in each round r (i.e., K := |K(r)|) where the sampling is uniform (without replacement) out of all N clients. We assume the communication and computation cost for a particular device in each round is the same, but varies among devices due to system heterogeneity. We do not consider the cost for model aggregation in Line 5, because it only needs to compute the average that is much less complex than local model updates.
>
> 2 Energy Cost: Similarly, by denoting ek as the per-round energy cost for client k to complete the computation and communication, we have
>
> ek = ek,pE + ek,m, (6)
>
> where ek,p and ek,m are  espectively the energy costs for client k to perform a local iteration and a round of communication.
>
> Our goal is to minimize the expected total cost while ensuring convergence.

Setup 1: Logistic regression and MNIST dataset, divide 6, 000 data samples (randomly sampled one-tenth of the total samples) among N = 20 Raspberry Pis in a noni.i.d. fashion with each device containing a balanced number of 300 samples of only 2 digits labels.

Setup 2: Simulated system using CNN and MNIST dataset, divide all 60, 000 data samples among N = 100 devices in the same non-i.i.d fashion as in Setup 1, but the amount of data in each
device follows the inherent unbalanced digit label distribution of MNIST, where the number of samples in each device has a mean of 600 and standard deviation of 20.1.

Setup 3: Simulated system using logistic regression and Synthetic (1, 1) dataset for statistical heterogeneity, where we generate 24, 517 data samples and distribute them among N = 100 devices in an unbalanced power law distribution, where the number of samples in each device has a mean of 245 and standard deviation of 362.

> We do not capture the energy cost in the prototype system because it is difficult to measure. For the simulation system, we generate the learning time and energy consumption for each client k using a normal distribution with mean tp = 0.1s, tm = 2s, ep = 10−3 J, and em = 2 × 10−2 J and standard deviation of the mean divided by 3. According to the definition of γ, we unify the time and energy costs such that one second is equivalent to 1−γ dollars (\$) and one Joule is equivalent to γ dollars (\$).

# Federated Learning over Wireless Networks: Optimization Model Design and Analysis
