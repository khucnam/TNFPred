# TNFPred
TNFPred
This is the public site for the paper under submission named: "TNFPred: Identifying tumor necrosis factors using hybrid features based on word embeddings"

Cytokines is a varied group of polypeptides, usually linked to inflammation and cell differentiation or death. Among major families of cytokines (interleukins (IL), interferons (IFNs), tumor necrosis factors (TNFs), chemokine and various growth factors, comprised of transforminggrowth factor b(TGF-b), fibroblast growth factor (FGF), heparin binding growth factor (HBGF) and neuron growth factor (NGF)) [1], tumor necrosis factors are versatile cytokines with a wide range of functions that attracts abundant of biological researchers (see, e.g. [2-6]). TNFs can take part in pathological reactions as well as involve in a variety of processes, such as inflammation, tumor growth, transplant rejection, etc. [3, 6]. TNFs act through their receptors at the cellular level to activate separate signals that control cell survival, proliferation or death. Furthermore, TNFs play two opposite roles in regard to cancer. On the positive side, activity in the suppression of cancer is supposed to be limited, primarily due to system toxicity of TNFs. On the negative side, TNFs might act as a promoter of the endogenous tumor through their intervention to the proliferation, invasion and tumor cell metastasis thus contributing to tumor provenance. Such TNFs' effect on cancer cell death makes them a probable therapeutic for cancer [3]. Moreover, in the United States and other nations, patients with TNF-linked autoimmune diseases have been authorized to be treated with TNF blockers [2]. In cytokine network, TNFs and other factors such as interleukins, interferons form an extremely complicated interactions generally mirroring cytokine cascades which begin with one cytokine causing one or additional different cytokines to express that successively trigger the expression of other factors and generate complex feedback regulatory circuits.  Abnormalities in these cytokines, their receptors, and the signaling pathways that they initiate involve a broad range of illnesses [7-12]. Interdependence between TNFs and other cytokines accounts for such diseases. For instance, TNFs and interleukin-1 administers TNF-dependent control of mycobacterium tuberculosis infection [12]. Another example is the TNF and type I interferons interactions in inflammation process which involve rheumatoid arthritis and systemic lupus erythematosus [13]. For the above reasons, identification of TNFs from other cytokines presents a challenge for many biologists. 

INSTRUCTION:

Using git bash to clone all the required files in "YOUR FOLDER" folder git clone https://github.com/khucnam/TNFPred

python Predict.py your_fasta_file.fasta ("your_fasta_file.fasta" file contains the sequences you want to classify. Please see the "sample.fasta" as an example.)

Running the Predict.py script will generate the Result.csv file. In Result.csv file, there are 2 columms: first one contains the protein ID, the next column contaisn a probability that a cytokine sequence is a TNF.