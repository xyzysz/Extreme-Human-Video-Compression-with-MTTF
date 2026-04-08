# [TCSVT2026]Extreme-Human-Video-Compression-with-Multi-granularity-Temporal-Trajectory-Factorization
### Shanzhi Yin&dagger;, Bolin Chen&dagger;, Shiqi Wang&dagger; and Yan Ye&sect;

#### &dagger; City University of Hong Kong and &sect; Alibaba Group

+ We propose a novel Multi-granularity Temporal Trajectory Factorization (MTTF) framework for extreme human video compression, which holds great potential for
bandwidth-constrained human-centered video streaming.
+ In particular, the proposed motion factorization strategy can facilitate to implicitly characterize the high-dimensional visual signal into compact motion vectors for representation compactness and further transform these vectors into a fine-grained field for motion expressibility. As such, the coded bit-stream can be entailed with enough visual motion information at the lowest representation cost. 
+ Meanwhile, a resolution-expandable generative module is developed with enhanced background stability, such that the proposed framework can be optimized towards higher reconstruction robustness and more flexible resolution adaptation.
+ Experimental results show that proposed method outperforms latest generative models and the state-of-the-art video coding standard Versatile Video Coding (VVC) on both talking-face videos and movingbody videos in terms of both objective and subjective quality.

![Fig 1](https://github.com/user-attachments/assets/7cf3b417-33d4-4631-8072-db166485a012)

## Subjective Demos

https://github.com/user-attachments/assets/25a7e8c3-52b3-4f40-b310-ab8be44cdad7

https://github.com/user-attachments/assets/e7c5c125-28e2-444a-aa79-ee25c38f19cc

https://github.com/user-attachments/assets/c1b7cef1-0a16-4b4c-b4d1-ab7c4050ee5e

https://github.com/user-attachments/assets/846c4c35-2d7c-4b37-adc0-854eebb6fbf1

https://github.com/user-attachments/assets/0b281538-8a2b-4eea-917a-ac467f7b6cda



## Code Release
Before running this code, download human matting model following [this link.](https://github.com/xyzysz/Extreme-Human-Video-Compression-with-MTTF/blob/main/SemanticGuidedHumanMatting/README.md)

For model training, excute
`run.py`

For encoding, excute
`encode.py`

For decoding, excute
`decode.py`

### :e-mail: Contact

If you have any question or collaboration need (research purpose or commercial purpose), please email `shanzhyin3-c@my.cityu.edu.hk`
