import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

TIMESTEPS = 20
NUM_BOARDS = 4
SENSORS = ['A', 'G']
AXES = ['x', 'y', 'z']

df = pd.read_csv("kicks_classification.csv")
#df = df.loc[0:100]


data = [[-0.2826, 10.5189, -1.1304, -0.9578, 0.4299, -0.2315, -0.0718, 9.4092, 1.7095, -0.116, 0.1586, -0.0255, 0.972, 9.2847, -2.4229, 0.1474, 0.0319, 0.0426, 1.0343, 9.9599, 4.2904, 0.5561, -0.0867, 0.2831], [0.0719, 10.1884, -0.7616, -0.5481, 0.6742, 0.1298, -0.8523, 9.4954, 2.7533, -0.1394, 0.1793, -0.1144, 1.1588, 8.983, -3.0215, 0.091, -0.0176, 0.1101, 1.0534, 11.4395, 6.6319, 1.1893, -0.4757, 0.2852], [2.4717, 9.992, -1.3508, 0.1522, 0.042, 0.1942, 0.905, 9.007, 3.0406, -0.3964, -0.0527, 0.05, 1.3886, 8.916, -2.8922, -0.0639, 0.1421, 0.0984, 1.8196, 9.1171, -1.6281, 1.325, -0.1256, -0.0085], [3.6596, 9.4172, -0.0431, 0.0628, -0.7035, 0.1463, 2.1691, 8.4324, 3.7014, -0.7285, -0.6417, 0.1357, 1.6951, 9.3278, -2.3894, -0.1878, 0.0777, 0.1038, -4.1899, 8.7245, -3.122, 0.3017, 0.158, -1.0397], [-1.3221, 10.3704, -1.2406, 0.2905, -0.1894, 0.0229, 2.3032, 7.3215, 4.8459, -1.143, -1.5181, 0.2788, 3.2609, 9.7731, -1.1396, -0.0862, -0.1953, 0.0404, 5.9376, 6.7373, 2.7629, -1.3904, 1.3995, -0.2224], [1.2646, 9.6615, -1.2789, 0.2996, -0.0649, 0.3006, 2.4373, 7.355, 7.1012, -2.4663, -1.566, 0.4576, 1.3934, 9.6917, -0.7949, 0.1059, -0.4087, 0.0394, 2.1117, 8.2696, 2.2984, -1.6996, -0.4337, -0.903], [2.1699, 9.4268, -0.843, 0.6604, -0.6018, 0.1149, -3.1891, 12.378, 2.8108, -0.6689, 0.5092, 0.0479, 2.3703, 9.6151, -1.6951, 0.307, -0.0575, -0.05, -0.0622, 9.3709, 1.8148, -2.0188, 0.1809, -0.5587], [1.2837, 9.3693, -0.685, 0.5859, -0.8551, 0.0011, 1.8866, 7.173, 3.5578, -0.3113, 0.2192, -0.5683, 0.905, 9.3565, -2.1835, 0.3209, -0.1091, -0.0314, 1.9824, 9.5481, 3.2609, -1.4622, -0.0681, -0.473], [2.0549, 9.3885, -1.2742, 0.4922, -0.986, 0.2022, 4.6878, 7.2257, 6.5457, 0.0724, -0.7636, -0.2315, 2.3846, 9.2081, -2.2505, 0.3272, -0.2123, -0.0245, -0.4836, 9.347, 1.6424, -0.7615, 0.5827, -0.3437], [2.1986, 11.2614, -1.6095, 0.6518, -0.5805, -0.282, 2.0686, 7.5369, 5.7221, 0.2804, -0.6832, -0.0809, 2.2697, 9.8497, -0.9481, 0.4954, -0.3182, -0.0867, 1.3743, 9.1506, 1.0056, -0.2469, 0.4097, -0.4294], [-0.4072, 10.4567, 0.5556, 0.3922, 0.0309, -0.7375, -0.0096, 6.6846, 7.5034, 0.0553, -0.1708, -0.3315, 1.6759, 9.7588, -1.2306, 0.7849, -0.4331, -0.1117, 2.241, 10.4244, 0.6704, 0.4161, -0.1086, -0.2895], [-1.643, 8.8759, 2.6297, -1.5788, 1.434, -0.307, 2.2218, 7.4316, 5.2625, -0.2214, 0.6327, -0.357, 2.7246, 10.0269, -3.3375, 0.9935, -0.4688, -0.2176, -0.8428, 8.3701, 6.7708, 0.016, 0.1016, -0.3746], [-5.0631, 4.6128, 2.5866, -4.7949, 2.7638, -0.9796, 4.0223, 10.0844, 5.703, 0.5827, 0.6188, -0.3544, -0.9625, 12.7994, -0.6273, 1.3122, -0.4837, -0.3209, 11.9375, 7.2113, 3.5243, 1.2467, -0.3677, -0.3459], [2.2944, 24.7597, 3.4153, -6.5301, 3.8908, -1.4223, 3.6488, 8.442, 6.6894, 0.8892, -0.042, -0.7295, 0.249, 14.1833, 3.5578, 2.1141, -0.2565, -0.3677, -10.4722, 8.0685, 0.9481, 1.4894, -3.1762, -0.2692], [-3.0369, 14.988, 2.2848, -5.2253, 2.3844, -1.6134, 3.6919, 12.311, 4.4053, 1.1408, -0.6109, -0.455, -2.0973, 17.3963, 5.9137, 3.2879, -0.4363, -0.4715, -3.2178, 10.635, 2.7294, 1.1196, 2.6856, -0.348], [-4.3254, 18.8728, 3.8416, -4.5879, 2.7665, -3.9637, -1.1636, 9.6534, 5.7604, 0.083, 0.0426, 0.0362, 3.6679, 24.349, 3.5817, 3.53, 0.5747, -0.1304, 2.375, 9.3901, 2.0782, 1.6639, 0.4741, 0.5497], [7.8844, 10.0639, 11.8889, -3.4476, 1.0291, -3.24, -7.2496, 5.6503, 4.3479, 0.1426, 0.5284, -0.1357, -0.091, 13.1633, -0.8092, 2.8394, 1.9002, -0.2714, -1.2737, 8.4803, -0.2011, 1.5011, 1.1621, 0.2554], [-13.8911, -7.5347, 13.3786, -7.4549, -3.7668, -3.6583, -2.8443, 6.79, 0.6656, -0.1426, 4.8523, -0.2607, 14.2359, 4.0127, -15.6054, -5.0641, -0.3475, 0.1799, 2.9209, 10.4244, 0.0958, 1.582, 1.3697, -0.1979], [-0.6275, 4.3685, -0.5796, -1.0243, 2.3189, -0.3837, -1.6807, 8.6957, 4.3144, -1.0169, 5.1642, 0.737, -5.7652, 22.1943, -4.5107, -11.5692, -3.7243, -3.8062, 3.6871, 5.679, -5.5306, 0.6417, -0.0287, -0.3618], [2.5483, 0.1198, -0.0671, -0.4257, -1.4399, 0.3038, 1.1827, 9.2273, 2.2218, -0.4257, 3.2555, 0.3214, -8.5425, 22.8359, -20.4561, -13.7615, 1.9002, -1.4452, 0.6512, 10.8314, -1.336, 0.2416, 1.209, -0.3437], [-8.2868, 2.2082, 3.7793, 3.0836, -1.0445, 0.7689, 0.3831, 9.5529, 1.3455, -0.6268, 1.0573, 0.216, 17.0563, 48.4108, 16.0747, -12.0672, 0.6396, -5.443, -0.0575, 9.7252, -1.1444, 0.6955, 1.309, -0.3629], [-4.8379, -0.4551, 7.5491, 6.1198, 2.6238, 3.0964, 0.4788, 9.8162, 3.6919, -0.687, 4.1068, -0.2426, -2.2122, 39.0494, 12.4163, -9.5152, -3.7551, -1.3159, -2.5762, 8.9016, -4.0318, 0.5608, 1.8821, -0.3954], [0.7999, 2.6681, 5.3888, 5.3004, 0.2804, 3.8615, 4.4676, 9.3039, 2.9353, -0.049, 1.6655, 0.6055, -14.461, 24.2581, -8.8825, -4.7225, 1.4096, -0.4933, -1.3024, 5.3056, -4.6064, 0.5635, 2.0178, -0.3342], [-16.3245, 11.4195, 11.3716, 4.1404, 0.2655, 3.073, -6.2058, 10.501, 9.4236, -0.0984, 0.9163, -0.2352, 15.6964, 3.213, -19.4696, -3.8222, 4.2936, -2.1923, 2.9401, 6.7133, -8.691, 0.3049, -0.7673, 0.2645], [-8.4257, 2.1268, 8.3921, 2.56, 1.8991, 4.2452, -11.0995, 3.6152, -1.7047, -0.895, 3.0038, 0.7902, 7.3167, 1.4078, -21.3084, -2.6462, -0.7641, -0.3374, -1.5897, 4.2042, -6.5457, 0.6194, -0.763, 0.3097], [-2.0214, 9.7621, 11.1464, 2.312, -2.4036, 3.7849, 12.6988, 13.2639, 5.0374, -0.2671, 2.6744, 1.1494, -1.2594, -3.7014, -23.0562, 0.3262, -1.8943, 0.7849, 3.5578, 5.3056, -7.4891, -0.1708, -0.3225, 0.0372], [-8.0473, 13.3019, 7.1659, 3.2134, -4.0776, 1.8299, -1.7573, 4.6687, 0.8284, 0.6119, -2.08, -0.4172, 1.5227, 0.1053, -16.4673, 3.1246, 0.729, 0.3948, 2.2266, 11.0947, -1.7142, -0.5295, -1.1988, 0.6864], [-8.2963, 7.7407, 3.3818, 2.4509, -1.7773, 0.7247, 0.182, 10.3094, -2.6097, 0.8008, -0.6577, -0.7577, -0.1437, 10.1275, -7.5465, 4.7087, 1.6772, 0.7434, -3.1029, 6.857, -0.6273, -0.5885, 0.0357, -0.0947], [-0.8622, 10.2651, 9.5274, 3.9249, -1.4766, 0.9546, 0.4022, 9.3087, 2.4421, 0.9434, -0.9988, 0.2208, 2.8012, 18.1959, 0.407, 5.5271, 1.8752, 1.8124, 1.1684, 8.0062, 2.3703, -0.2538, -1.2297, 0.4012], [-3.765, -0.6131, 2.0501, 1.8677, -1.3516, -0.0074, -0.3448, 9.6343, 2.7629, 0.7508, -1.6644, 0.3459, -0.3639, 23.8079, 7.6519, 5.3552, -0.1469, 2.9511, 3.371, 2.2697, -15.2367, 1.5469, 1.8438, 0.0479], [-0.2299, 13.2924, 5.2834, -1.9539, 0.8226, 0.3847, -4.4005, 6.7852, 3.9887, 0.7322, -2.1481, 0.3464, -9.3374, 29.961, 14.4179, 6.6791, 0.4661, 2.8622, -4.707, 11.1091, -7.4747, -0.257, 0.3017, 0.0851], [-4.2344, 8.9957, -1.4514, -1.4372, 1.6075, -0.1846, -2.5714, 8.983, 10.2328, 0.2613, -2.5637, 0.0527, -11.473, 24.2964, 14.8393, 7.5188, 2.8894, 2.038, 2.0686, 7.6184, -6.948, -0.0686, -0.3065, 0.1431], [0.7664, 9.4555, -0.297, -0.3336, -0.4566, 0.0963, 2.6719, 6.8043, 2.5426, 0.4358, -3.737, 1.4926, -3.5434, 22.4145, -0.067, 3.6684, 0.3469, 2.287, -3.1268, 9.1171, -5.3486, -0.7301, -0.4661, 0.158], [-0.7904, 9.3166, -0.8718, -0.1697, 0.6268, 0.0548, -14.5472, 11.157, 6.7564, -0.8636, 2.9963, -2.3525, -10.5776, 19.7139, 9.5433, 2.7255, -0.9296, 1.1238, 1.336, 10.0748, -3.9217, -0.9865, -1.3521, 0.1852], [0.4263, 9.4507, -0.6658, -0.1431, 0.1862, -0.0069, 19.5271, 13.4985, -4.2377, 1.0977, -2.6302, 1.1036, -2.4038, 10.3238, 2.8491, 0.7396, -0.5614, 0.8237, -1.494, 8.9543, 0.5171, -0.5954, 0.2959, -0.0447], [-0.1389, 9.4651, -0.7999, 0.0074, 0.2522, 0.0298, -11.6885, 11.5975, 13.1825, 2.1029, -3.1458, -0.8359, -3.0502, 11.1282, 2.5666, -0.5912, -1.0786, 0.3539, -0.5076, 9.5624, -1.7047, -0.3155, 0.1453, 0.0729], [-0.4072, 9.4939, -0.5269, -0.074, -0.0612, -0.0862, -2.6624, 6.7133, 0.2059, 0.7444, -1.1408, -0.2613, 0.0335, 23.2764, 10.7356, -3.5646, -0.92, 0.5321, -0.5171, 8.2025, -3.4237, -0.3704, 0.0266, 0.1006], [-0.9388, 9.6376, -0.3928, -0.3166, 0.0756, -0.1282, -1.2258, 7.3454, 1.5706, 1.6166, 0.7215, 0.5305, -2.897, 10.477, 20.7769, -0.6263, 1.931, 0.1517, 1.0199, 9.2081, -2.3846, -0.4667, -0.0351, 0.0234], [-0.5317, 9.6759, -0.5269, 0.0357, 0.1458, -0.0011, -1.7669, 7.9009, 1.4748, 0.2357, -0.1532, -0.1405, -2.4804, 8.3845, 1.0295, 0.7955, 1.3702, 0.6247, -1.0199, 9.2847, -0.9864, -0.2804, 0.4262, 0.075], [-0.6562, 9.6807, -0.3832, 0.0133, 0.0506, -0.0027, -1.4892, 10.7452, 0.3208, 0.4986, 0.1554, -0.3666, -2.9928, 8.758, -4.0606, 0.8083, 1.3149, 0.2379, -0.0192, 10.1323, -1.221, -0.1894, 0.6965, 0.0937], [-0.4598, 9.8675, -0.5652, -0.0436, 0.0367, -0.0692, 0.4501, 9.596, -1.4844, 0.1634, 0.2458, 0.1974, 0.4836, 9.8402, -0.0287, 0.0708, 0.1298, -0.049, 0.5219, 9.6774, -1.968, -0.025, 0.0016, -0.0372], [-0.3401, 9.9202, -0.7425, -0.0511, 0.1192, -0.0665, 0.0814, 10.2759, -1.561, 0.2251, 0.1862, -0.0644, -0.0383, 9.6726, -0.7661, 0.0553, 0.0367, -0.05, 0.7135, 9.6199, -2.0877, -0.0926, -0.0053, -0.058], [-0.2299, 9.6184, -0.4119, -0.0176, 0.116, -0.0553, -0.6369, 9.0788, -0.8092, 0.2469, 0.0436, -0.133, 0.4357, 9.5529, -0.905, 0.009, 0.0878, -0.0697, 0.6081, 9.6247, -1.7286, -0.1038, 0.0287, -0.0553], [-0.9724, 9.7094, -0.3401, -0.0282, 0.1277, -0.0915, -0.3496, 8.9878, -1.0343, -0.1681, 0.1394, 0.0623, -0.2634, 9.6247, -0.5076, -0.0245, 0.1405, -0.0543, 0.4453, 9.7444, -1.6041, -0.0984, 0.0069, -0.0516], [-0.3976, 9.8387, -0.3353, -0.2799, -0.1075, -0.1059, 0.0766, 9.8785, -0.8906, -0.2554, 0.0931, 0.0968, 0.4932, 9.6103, -0.5124, 0.0074, 0.1362, -0.067, 0.4501, 9.6103, -1.6807, -0.0809, 0.0495, -0.0341], [-0.8095, 9.7286, 0.1581, -0.1602, -0.1176, -0.1059, -0.1245, 9.7205, -1.3264, 0.0761, -0.0229, -0.0921, 0.1484, 9.6534, -0.4788, 0.0059, 0.0415, -0.0734, 0.7518, 9.6582, -1.7957, -0.0867, 0.0186, -0.0532], [-1.0107, 9.7334, 0.3305, -0.1293, -0.1761, -0.1, -0.4836, 9.4332, -1.3982, 0.0612, -0.0867, -0.0633, -0.0144, 9.6103, -0.6991, 0.0032, 0.0644, -0.0851, 0.6273, 9.6247, -1.7382, -0.1059, -0.0277, -0.0601], [-0.7281, 9.6711, 0.9772, -0.2214, -0.2618, -0.1235, -0.565, 9.2129, -1.1588, -0.0639, 0.0069, 0.0176, 0.1149, 9.6247, -0.8284, -0.0367, 0.0921, -0.0718, 0.5411, 9.6391, -1.7909, -0.1091, -0.0681, -0.083], [-1.4753, 9.9441, 1.5999, -0.0724, -0.4486, -0.0963, -0.3256, 9.7205, -1.0774, -0.1719, 0.0442, -0.0319, 0.2634, 9.6007, -0.5363, -0.0548, 0.0766, -0.074, 0.4022, 9.6007, -1.9393, -0.1383, 0.0208, -0.0798], [-2.2848, 9.6615, 1.8825, 0.1219, -0.1596, -0.0745, -0.1963, 9.5768, -1.1732, -0.099, -0.0133, -0.0564, -0.0144, 9.6007, -0.5171, -0.0474, 0.017, -0.074, 0.4214, 9.5624, -1.5562, -0.1681, 0.1224, -0.0931], [-1.9591, 9.1394, 1.5424, 0.232, 0.4156, -0.1389, -0.3352, 9.4044, -1.0104, -0.0873, -0.0282, -0.0053, 0.1149, 9.5337, -0.5746, -0.0404, 0.0431, -0.0867, 0.4405, 9.5385, -1.6951, -0.174, 0.1112, -0.0931], [-0.3545, 9.6328, 1.4993, 0.0745, 0.3166, -0.1889, 0.0862, 9.3518, -1.0726, -0.2506, -0.1485, 0.0857, -0.0622, 9.6869, -0.5794, -0.067, 0.0564, -0.0867, 0.5698, 9.6247, -1.5036, -0.1916, 0.0649, -0.1086], [-2.9171, 8.737, 3.2955, -0.5672, 0.3203, -0.0325, -0.7709, 10.0796, -1.494, 0.0463, 0.0468, -0.0202, 0.2825, 9.6055, -0.4693, -0.0809, 0.0245, -0.1075, 0.316, 9.6007, -1.3791, -0.2054, 0.0686, -0.1298], [-2.2848, 4.4691, 5.4894, -2.353, 0.4523, -0.4422, -0.1293, 9.3565, -0.9912, -0.133, -0.0436, -0.1181, -0.0862, 9.6343, -0.3974, -0.0814, 0.0559, -0.1197, 0.5267, 9.6678, -1.3838, -0.2118, 0.0048, -0.149], [-16.0802, 14.1977, 6.2462, -6.1603, 2.8074, -2.2115, -0.5555, 9.481, -1.1396, -0.116, 0.1878, -0.0591, -0.2346, 9.7205, -0.2059, -0.058, 0.1325, -0.1352, -0.0527, 9.73, -0.929, -0.1687, -0.0064, -0.1586], [5.9253, 24.5729, 8.4017, -3.9658, 3.5226, -3.3172, -0.4932, 9.4619, -0.9529, -0.1575, 0.1511, -0.0314, 0.0622, 9.73, -0.2251, -0.0415, 0.141, -0.1426, -0.2969, 9.6391, -1.5179, -0.1421, 0.2299, -0.1532], [-8.349, 2.7207, 2.9842, -3.5965, -1.119, -5.9825, 0.0575, 9.4475, -0.7805, -0.281, 0.0617, 0.0043, 0.1293, 9.6869, -0.565, -0.0452, 0.05, -0.1719, 1.063, 9.6678, -0.2825, -0.2474, -0.075, -0.1692], [2.5962, -5.7145, -0.7472, -2.71, -0.4331, -2.1577, 0.0144, 9.7348, -0.7374, -0.1288, -0.191, -0.1203, 0.1484, 9.5672, -0.6991, -0.0788, -0.1128, -0.1857, -0.6369, 9.8258, -0.0575, -0.1133, -0.2123, -0.1799], [-7.1515, 3.6165, 0.9149, -1.5011, 0.8647, -0.2261, 0.0192, 9.5912, -0.5459, -0.1282, -0.2703, -0.0053, -0.4884, 9.5816, -0.6081, -0.1554, -0.1788, -0.1926, -0.2059, 9.7827, -0.68, -0.0686, -0.2054, -0.1352], [1.6813, -5.1014, 0.7664, 0.3852, 0.1431, 0.3017, -0.5076, 9.6007, -0.0431, -0.0431, -0.3139, -0.0005, -0.34, 9.6917, -0.1053, -0.1469, -0.2107, -0.1777, -1.0247, 9.9455, -0.7661, 0.0223, -0.0372, -0.1064], [-6.7061, 0.7808, 3.0465, 2.4961, 0.0351, 1.6522, -0.6033, 9.596, 0.6177, 0.0676, -0.323, -0.0761, -0.407, 9.7396, 0.1006, -0.1309, -0.298, -0.1767, -1.2402, 9.9407, -0.5602, 0.066, 0.2139, -0.0436], [-8.3011, -0.3976, 8.0664, 4.0334, 1.8204, 3.3108, 0.1245, 9.8497, 0.1676, 0.0128, -0.5534, -0.0101, -0.6704, 9.7492, 0.3304, -0.0734, -0.3363, -0.1602, 0.1437, 10.0269, -0.1101, 0.1857, 0.5401, -0.008], [-9.5082, 9.2927, 8.555, 2.7505, -2.9053, 2.2333, -0.8428, 9.8306, 0.6416, -0.0521, -0.7253, -0.075, -1.3216, 9.6917, 0.3112, -0.0128, -0.2086, -0.1282, 0.9529, 9.8689, 0.4884, 0.3443, -0.0649, -0.017], [-6.7396, 8.9478, -0.2539, 2.4419, -1.3952, 0.9349, -1.6807, 9.3757, 0.0575, 0.033, -0.1618, -0.0607, -1.0008, 10.3765, 0.4597, 0.1235, 0.1197, -0.1059, -1.0726, 10.1754, 1.8004, 0.588, -0.7418, 0.0543], [-0.9197, 12.7128, 1.164, 0.9833, -1.3585, 0.4997, -1.1444, 9.7252, 0.1628, -0.0309, 0.1735, -0.0511, -1.5658, 9.845, 1.221, 0.3038, 0.2783, -0.0617, 1.5562, 8.8777, 1.5658, 1.0759, -0.2948, 0.0835], [-5.2403, 3.8704, 4.2919, 0.2363, -0.7849, -0.1793, -0.3112, 9.7109, 0.2873, -0.1612, 0.2927, -0.2336, -0.0383, 10.1514, 0.5794, 0.4613, 0.448, -0.0809, -2.3703, 10.0748, -3.5099, 1.2366, -0.2905, 0.3022], [0.9245, 11.0458, -0.2443, 0.8147, -0.7082, 0.1107, -0.4741, 9.4858, 0.8954, -0.0995, 0.1777, -0.0357, -0.2442, 10.477, 0.8906, 0.6359, 0.1846, -0.0591, -1.7909, 9.3039, -6.0765, 0.9945, 0.8881, -0.0926], [-0.7808, 11.0842, 1.437, -0.1905, 0.1096, 0.1442, -0.6464, 9.0549, 1.968, -0.1192, -0.3251, 0.0202, -2.2027, 12.378, 7.2257, 1.4942, -0.0074, 0.0399, 6.5314, 14.7291, -0.3352, 0.5156, 1.1962, -0.166], [-1.0969, 9.719, 0.9436, 0.1878, -0.0718, -0.1378, -2.9161, 7.6184, 0.1628, -1.0121, 0.4933, -0.216, -3.055, 18.5311, 10.5824, 2.4983, 0.1394, 0.3549, -0.3065, 12.1721, -1.6424, 0.7412, 1.2749, -0.306], [-0.8478, 9.4268, -0.1437, 0.4262, 0.4895, 0.2528, -7.1347, 7.7381, -0.8763, -2.1242, 1.7852, 0.5619, -6.0765, 31.7136, 1.7957, 2.5909, -1.4841, 0.3118, 5.6695, 1.1013, -1.0918, -0.3757, 0.2379, -0.1336], [-0.2012, 9.3454, -0.1437, 0.165, 0.2464, 0.132, 3.9744, 12.6988, 0.5076, -1.7262, 3.4816, 0.9775, -6.3351, 9.8737, 2.3846, -0.787, -0.9535, 0.0718, -6.4356, 10.0269, -5.1284, -0.3485, -0.6353, -0.422], [-0.4359, 8.8376, -0.4167, 0.1378, 0.1181, 0.0069, 2.4325, 11.9519, 4.0223, -1.0158, 0.8258, -0.7779, -6.1962, 27.7201, -15.9597, -4.5075, -6.9446, 1.019, 1.9441, 9.754, -1.0534, 0.1825, 2.0156, -0.6407], [-0.7425, 9.2065, 0.5892, 1.0685, 0.5114, 0.183, -3.2274, 7.7524, 4.4436, -1.8858, 2.179, -0.6566, -1.5036, 21.701, 5.9999, -13.8392, 0.0378, -2.6132, 1.4317, 7.1251, -3.9935, 0.3586, 1.7427, -0.1086]]

def classify_sample(registrazione_smv, threshold=7.0):
    """
    Classifica una registrazione come 'Calcio' o 'Fermo' basandosi su una semplice soglia.

    :param registrazione_smv: Una lista o array di 8 valori SMV.
    :param soglia: Il valore di SMV oltre il quale si considera un calcio.
    :return: La stringa 'Calcio' o 'Fermo'.
    """
    # Controlla se QUALSIASI valore nella lista supera la soglia
    if np.mean(registrazione_smv) > threshold:
        return "Calcio"
    
    # Se il ciclo finisce senza aver trovato valori sopra la soglia
    return "Fermo"

def compute_smv(df):
    out_list = []
    in_array = df.to_numpy()
    
    for row in in_array:
        temp_list = []
        for column in range(0, len(row), 3):
            smv = np.sqrt(row[column]**2 + row[column+1]**2 + row[column+2]**2)
            temp_list.append(smv)
        out_list.append(temp_list)
    
    return np.array(out_list)

def generate_column_names():
    column_names = []
    for i in range(TIMESTEPS):
        suffix = f"_{i + 1}"
        for board_id in range(1, NUM_BOARDS + 1):
            for sensor_type in SENSORS:
                for axis in AXES:
                    col_name = f"{sensor_type}{board_id}{axis}{suffix}"
                    column_names.append(col_name)
    return column_names

'''
lista2 = []
for i in range(len(compute_smv(df)[0:50])):
    array = compute_smv(df)[i]
    counter = 0
    counter2 = 0
    lista = []
    for j in range(len(array)):
        if counter % 8 == 0:
            start = counter2 * 8
            end = (counter2 + 1) * 8
            lista = array[start:end]
            counter2 += 1 
            lista2.append(lista)
            #print(f"Questa è la registrazione numero {(counter // 8) + 1}")
            #print(lista)
            #print(classifica_con_soglia(lista, soglia=14.0))
        counter += 1

#print(np.array(lista2))

lista3 = np.array([])
for i in range(len(np.array(lista2))):
    lista3 = np.append(lista3, classifica_con_soglia(lista2[i]))
'''
def crea_gruppi_(dati_input, dimensione_gruppo=20, min_calcio_len=3):
    """
    Crea gruppi di dimensione fissa garantita, senza sovrapposizioni di indici.
    I gruppi sono centrati solo su blocchi di "Calcio" con una lunghezza minima.

    Args:
        dati_input (np.ndarray): L'array 1D con etichette "Calcio" e "Fermo".
        dimensione_gruppo (int): La dimensione esatta di ogni gruppo.
        min_calcio_len (int): La lunghezza minima di una sequenza di "Calcio"
                              per poter essere considerata come centro di un gruppo.

    Returns:
        list: Una lista di liste, dove ogni lista interna è un gruppo di tuple
              (etichetta, indice_originale).
    """
    
    # 1. Identifica tutti i blocchi (Fermo e Calcio)
    blocchi = []
    if len(dati_input) == 0: return []
    label_corrente, start_index = dati_input[0], 0
    for i in range(1, len(dati_input)):
        if dati_input[i] != label_corrente:
            blocchi.append({'label': label_corrente, 'start': start_index, 'end': i})
            label_corrente, start_index = dati_input[i], i
    blocchi.append({'label': label_corrente, 'start': start_index, 'end': len(dati_input)})
    
    # 2. Filtra per tenere solo i blocchi di Calcio che superano la lunghezza minima
    blocchi_calcio_validi = [
        b for b in blocchi 
        if b['label'] == 'Calcio' and (b['end'] - b['start']) >= min_calcio_len
    ]

    gruppi_finali = []
    ultimo_indice_usato = -1

    # 3. Itera sui blocchi di Calcio validi per costruire i gruppi sequenzialmente
    for calcio_block in blocchi_calcio_validi:
        calcio_start = calcio_block['start']
        
        # Se questo blocco di Calcio è già stato "consumato" dal padding del gruppo precedente, saltalo
        if calcio_start <= ultimo_indice_usato:
            continue

        calcio_end = calcio_block['end']
        num_calcio = calcio_end - calcio_start

        if num_calcio >= dimensione_gruppo:
            start_gruppo, end_gruppo = calcio_start, calcio_end
        else:
            padding_necessario = dimensione_gruppo - num_calcio
            pre_padding_target = padding_necessario // 2
            post_padding_target = padding_necessario - pre_padding_target

            # Calcola i limiti per il prelievo del padding
            # Il padding iniziale può essere preso solo da dopo l'ultimo gruppo creato
            limite_pre = ultimo_indice_usato + 1
            # Il padding finale può essere preso fino alla fine dell'array
            limite_post = len(dati_input)

            # Calcola il padding disponibile e prelevalo
            pre_padding_disponibile = calcio_start - limite_pre
            post_padding_disponibile = limite_post - calcio_end

            pre_da_prendere = min(pre_padding_disponibile, pre_padding_target)
            post_da_prendere = min(post_padding_disponibile, post_padding_target)
            
            # Logica di compensazione
            mancanti = padding_necessario - (pre_da_prendere + post_da_prendere)
            if mancanti > 0:
                extra_post = min(mancanti, post_padding_disponibile - post_da_prendere)
                post_da_prendere += extra_post
                mancanti -= extra_post
            if mancanti > 0:
                extra_pre = min(mancanti, pre_padding_disponibile - pre_da_prendere)
                pre_da_prendere += extra_pre
            
            # Se ancora mancano elementi, questo gruppo non può essere formato a 20.
            if pre_da_prendere + post_da_prendere + num_calcio < dimensione_gruppo:
                print(f"ATTENZIONE: Blocco 'Calcio' (indici {calcio_start}-{calcio_end-1}) non ha abbastanza elementi circostanti per formare un gruppo di {dimensione_gruppo}. Saltato.")
                continue

            start_gruppo = calcio_start - pre_da_prendere
            end_gruppo = calcio_end + post_da_prendere

        # Estrai il gruppo e aggiorna l'ultimo indice usato
        indici_gruppo = np.arange(start_gruppo, end_gruppo)
        etichette_gruppo = dati_input[indici_gruppo]
        gruppo = list(zip(etichette_gruppo, indici_gruppo))
        gruppi_finali.append(gruppo)
        
        ultimo_indice_usato = end_gruppo - 1
        
    return np.array(gruppi_finali)

def generate_columns():
        column_names = []
        for board_id in range(1, NUM_BOARDS + 1):
            for sensor_type in SENSORS:
                for axis in AXES:
                    col_name = f"{sensor_type}{board_id}{axis}"
                    column_names.append(col_name)
        return column_names

def crea_dataframe_da_gruppi(etichette, df_dati, dimensione_gruppo=20, min_calcio_len=3):
    """
    Crea un DataFrame finale (es. 480 colonne) partendo da un array di etichette e
    un DataFrame di misurazioni.

    La logica di raggruppamento garantisce che i gruppi abbiano sempre una dimensione fissa,
    siano centrati su blocchi di "Calcio" di lunghezza minima e non usino mai
    gli stessi indici in più di un gruppo.

    Args:
        etichette (np.ndarray): L'array 1D con le etichette "Calcio" e "Fermo".
        df_dati (pd.DataFrame): Il DataFrame originale con le misurazioni (es. 24 colonne).
        dimensione_gruppo (int): La dimensione esatta (in timestep) di ogni gruppo.
        min_calcio_len (int): La lunghezza minima di una sequenza di "Calcio"
                              per poter formare un gruppo.

    Returns:
        pd.DataFrame: Un nuovo DataFrame in cui ogni riga rappresenta un gruppo
                      appiattito (es. 20 * 24 = 480 colonne).
    """
    
    # 1. Identifica tutti i blocchi (Fermo e Calcio)
    blocchi = []
    if len(etichette) == 0:
        return pd.DataFrame()
        
    label_corrente, start_index = etichette[0], 0
    for i in range(1, len(etichette)):
        if etichette[i] != label_corrente:
            blocchi.append({'label': label_corrente, 'start': start_index, 'end': i})
            label_corrente, start_index = etichette[i], i
    blocchi.append({'label': label_corrente, 'start': start_index, 'end': len(etichette)})
    
    # 2. Filtra per tenere solo i blocchi di Calcio che superano la lunghezza minima
    blocchi_calcio_validi = [
        b for b in blocchi 
        if b['label'] == 'Calcio' and (b['end'] - b['start']) >= min_calcio_len
    ]

    lista_gruppi_appiattiti = []
    ultimo_indice_usato = -1

    # 3. Itera sui blocchi validi per costruire i gruppi senza sovrapposizioni
    for calcio_block in blocchi_calcio_validi:
        calcio_start = calcio_block['start']
        
        if calcio_start <= ultimo_indice_usato:
            continue

        calcio_end = calcio_block['end']
        num_calcio = calcio_end - calcio_start

        # 4. Logica per garantire che il gruppo sia SEMPRE di dimensione fissa
        if num_calcio >= dimensione_gruppo:
            # Se il blocco è troppo grande, prendiamo solo i primi `dimensione_gruppo` elementi
            start_gruppo = calcio_start
            end_gruppo = calcio_start + dimensione_gruppo
        else:
            # Altrimenti, calcola il padding necessario
            padding_necessario = dimensione_gruppo - num_calcio
            pre_padding_target = padding_necessario // 2
            
            # Limiti da cui possiamo prelevare padding
            limite_pre = ultimo_indice_usato + 1
            limite_post = len(etichette)

            pre_padding_disponibile = calcio_start - limite_pre
            post_padding_disponibile = limite_post - calcio_end
            
            # Logica di compensazione per distribuire il padding
            pre_da_prendere = min(pre_padding_disponibile, pre_padding_target)
            # Calcola quanto serve dopo, tenendo conto di quanto abbiamo già preso prima
            post_da_prendere = min(post_padding_disponibile, padding_necessario - pre_da_prendere)
            
            # Se dopo non c'era abbastanza, prova a prendere il resto da prima
            mancanti = padding_necessario - (pre_da_prendere + post_da_prendere)
            if mancanti > 0:
                pre_da_prendere += min(mancanti, pre_padding_disponibile - pre_da_prendere)
            
            # Se ancora non si raggiunge la dimensione, il gruppo non può essere formato
            if pre_da_prendere + post_da_prendere + num_calcio < dimensione_gruppo:
                continue

            start_gruppo = calcio_start - pre_da_prendere
            end_gruppo = calcio_end + post_da_prendere

        # 5. Estrai la fetta dal DataFrame, appiattiscila e salvala
        gruppo_df = df_dati.iloc[start_gruppo:end_gruppo]
        riga_appiattita = gruppo_df.values.flatten()
        lista_gruppi_appiattiti.append(riga_appiattita)
        
        ultimo_indice_usato = end_gruppo - 1
        
    if not lista_gruppi_appiattiti:
        return pd.DataFrame()

    return pd.DataFrame(lista_gruppi_appiattiti)


kick_df = pd.DataFrame(data, columns=generate_columns())
smv_raw_output = compute_smv(kick_df)
smv_classified = np.array([])
for i in range(len(np.array(smv_raw_output))):
    smv_classified = np.append(smv_classified, classify_sample(smv_raw_output[i], threshold=6.5))

print(crea_dataframe_da_gruppi(smv_classified, kick_df))




'''
smv = []
counter = 0
for j in range(len(smv_raw_output)):
    if j % 8 == 0:
        start = counter * 8
        end = (counter + 1) * 8
        chunk = smv_raw_output[start:end]
        counter += 1 
        smv.append(chunk)

print(len(smv))'''
#for i in range(len(smv)):
    #print(smv[i])

#print(len(smv))
'''
smv_classified = np.array([])
for i in range(len(np.array(smv))):
    smv_classified = np.append(smv_classified, self.classify_sample(smv[i], threshold=6.5))

#print(smv_classified)

#print(len(smv_classified))
#if len(smv_classified) != len(df):
    #print('AAAAAAAAAAAAAAAAAAAAA')

gruppi = self.crea_gruppi_(smv_classified, df)'''