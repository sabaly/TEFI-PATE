# TEFI-PATE
*** TEFI-PATE for Teacher ensemble Fairness Impact - PATE ***

# Introduction 
This project contains the experiments conducted during my internship research on <i>Fairness and Confidentiality in PATE</i>. Our contributions are as follows:

1. Through an empirical analysis of the PATE framework, we evaluate the fairness of
the student model with respect to the fairness levels of the teachers’ ensemble. We
show that the fairness of the student model improves significantly when there is a
high proportion of fair teachers in the ensemble.

2. We propose a fairness-aware voting mechanism that weights (via duplication) the
teachers’ votes proportionally to their fairness metrics. We empirically demonstrate
the positive impact of this voting mechanism on student model fairness using two
state-of-the-art group fairness metrics.

3. We analyse the security of the proposed approach in an honest but curious adversarial
setting. Specifically, we observe that fairness measures of teachers’ models strongly
indicate vulnerability to an attribute inference attack.

4. We propose a private and fairness-aware voting mechanism that leverages fully homomorphic encryption. Specifically, we use the CKKS scheme to encrypt the
entire labeling process.

5. Finally, we conduct extensive experiments using the ACSEmployement dataset from
the folktables collection and the Adult Income dataset. We utilize the CKKS FHE
scheme for encrypted inference, applying different polynomial approximations
to the activation functions of our teacher models. For this, we use the LATTIGO
package, which provides a CKKS implementation along with various functionalities.

This README provides guidelines on how to use this repository. To compile some of the files, you may need the checkpoints (folders storing teacher models) available in my Google Drive: https://drive.google.com/file/d/1AEoGBt1tEGemBkKTpXS7kPYXsbdZAEEr/view?usp=drive_link. Place them in the TEFI-PATE/ folder.

# Teacher Ensemble fairness impact experiments
Experiment 01: The relevant files are ```src/fairnessimpact_acsemployment.py``` and ```src/fairnessimpact_adult.py```. These files generate curves showing how student fairness varies based on the number of fair teachers in the teacher ensemble. They can also be adapted to compare student fairness using standard aggregation versus weighted aggregation. 

Experiment 02:  The relevant files are ```src/acsemployment_beta_impact.py``` and ```src/adult_beta_impact.py```. This experiment demonstrates the impact of the $\beta$ parameter in our weight computation function.


