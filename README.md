<div align="center">
<h1> 🔥CM-CGNS🎉 </h1>
<h3>Cross-Modal Clustering-Guided Negative Sampling for Self-Supervised Joint Learning from Medical Images and Reports</h3>

[Libin Lan](https://orcid.org/0000-0003-4754-813X)<sup>1</sup> ,[Hongxing Li](https://orcid.org/0009-0002-7958-3976)<sup>1</sup> ,[Zunhui Xia](https://orcid.org/0009-0008-6706-5817)<sup>1</sup> ,[Juan Zhou](https://orcid.org/0009-0008-0243-3949)<sup>2</sup> ,[Xiaofei Zhu](https://orcid.org/0000-0001-8239-7176)<sup>1</sup>,[Yongmei Li](https://orcid.org/0000-0003-2829-6416)<sup>3</sup> ,[Yudong Zhang](https://orcid.org/0000-0002-4870-1493)<sup>4</sup>,[Xin Luo](https://orcid.org/0000-0002-1348-5305)<sup>5 :email:</sup>

🏢 <sup>1</sup> College of Computer Science and Engineering, Chongqing University of Technology.
🏢 <sup>2</sup> Department of Pharmacy, the Second Affiliated Hospital of Army Military Medical University.
🏢 <sup>3</sup> Department of Radiology, the First Affiliated Hospital of Chongqing Medical University.
🏢 <sup>4</sup> School of Computer Science and Engineering, Southeast University.
🏢 <sup>5</sup> College of Computer and Information Science, Southwest University,  (<sup>:email:</sup>) corresponding author.
</div>

## 👇Overview
  
### • Abstract
Learning medical visual representations directly from paired images and reports through multimodal self-supervised learning has emerged as a novel and efficient approach to digital diagnosis in recent years. However, existing models suffer from several severe limitations. 1) neglecting the selection of negative samples, resulting in the scarcity of hard negatives and the inclusion of false negatives; 2) focusing on global feature extraction, but overlooking the fine-grained local details that are crucial for medical image recognition tasks; and 3) contrastive learning primarily targets high-level features but ignoring low-level details which are essential for accurate medical analysis. Motivated by these critical issues, this paper presents a Cross-Modal Cluster-Guided Negative Sampling (CM-CGNS) method with two-fold ideas. First, it extends the k-means clustering used for local text features in the single-modal domain to the multimodal domain through cross-modal attention. This improvement increases the number of negative samples and boosts the model representation capability. Second, it introduces a Cross-Modal Masked Image Reconstruction (CM-MIR) module that leverages local text-to-image features obtained via cross-modal attention to reconstruct masked local image regions. This module significantly strengthens the model's cross-modal information interaction capabilities and retains low-level image features essential for downstream tasks. By well handling the aforementioned limitations, the proposed CM-CGNS can learn effective and robust medical visual representations suitable for various recognition tasks. Extensive experimental results on classification, detection, and segmentation tasks across five downstream datasets show that our method outperforms state-of-the-art approaches on multiple metrics, verifying its superior performance.


## Once our paper is accepted, we will release the code as soon as possible.
