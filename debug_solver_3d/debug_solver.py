import math
import numpy as np
import iric
import sys
import os

print("----------Start----------")

###############################################################################
# CGNSを開く
###############################################################################

# iRICで動かす時用
# =============================================================================
if len(sys.argv) < 2:
    print("Error: CGNS file name not specified.")
    exit()

cgns_name = sys.argv[1]

print("CGNS file name: " + cgns_name)

# # CGNSをオープン
fid = iric.cg_iRIC_Open(cgns_name, iric.IRIC_MODE_MODIFY)

# コマンドラインで動かす時用
# =============================================================================

# CGNSをオープン
# fid = iric.cg_iRIC_Open("./project/Case1.cgn", iric.IRIC_MODE_MODIFY)

# 分割保存したい場合はこれを有効にする
# os.environ['IRIC_SEPARATE_OUTPUT'] = '1'

###############################################################################
# 古い計算結果を削除
###############################################################################

iric.cg_iRIC_Clear_Sol(fid)

###############################################################################
# 計算条件を読み込み
###############################################################################

# 格子サイズを読み込み
isize, jsize = iric.cg_iRIC_Read_Grid2d_Str_Size(fid)

# 格子点の座標読み込み
# --------------------------------------------------
# メモ
# --------------------------------------------------
# CGNSから読み込む時は1次元配列、順番は以下
# --------------------------------------------------
#      j
#      ↑
#     4| 24, 25, 26, 27, 28, 29
#     3| 18, 19, 20, 21, 22, 23
#     2| 12, 13, 14, 15, 16, 17
#     1|  6,  7,  8,  9, 10, 11
#     0|  0,  1,  2,  3,  4,  5
#       ----------------------- →　i
#         0   1   2   3   4   5
# --------------------------------------------------
grid_x_arr_2d, grid_y_arr_2d = iric.cg_iRIC_Read_Grid2d_Coords(fid)
grid_x_arr_2d = grid_x_arr_2d.reshape(jsize, isize).T
grid_y_arr_2d = grid_y_arr_2d.reshape(jsize, isize).T

# 計算時間を読み込み
time_end = iric.cg_iRIC_Read_Integer(fid, "time_end")
ksize = iric.cg_iRIC_Read_Integer(fid, "z_division")+1
z_height = iric.cg_iRIC_Read_Real(fid, "z_height")

# 読み込んだ格子サイズをコンソールに出力
print("Grid size:")
print("    isize= " + str(isize))
print("    jsize= " + str(jsize))
print("    ksize= " + str(ksize))

###############################################################################
# 格子をz方向に拡張
# x, y座標は読み込んだものをそのまま使う、z座標は等間隔で追加0~z_height
###############################################################################

# 2次元のx, y座標を3次元に拡張[isize, jsize, ksize]
grid_x_arr_3d = grid_x_arr_2d[:, :, np.newaxis] * np.ones(ksize)
grid_y_arr_3d = grid_y_arr_2d[:, :, np.newaxis] * np.ones(ksize)
grid_z_arr_3d = np.linspace(0, z_height, ksize)[np.newaxis, np.newaxis, :]
grid_z_arr_3d = np.broadcast_to(grid_z_arr_3d, (isize, jsize, ksize))

###############################################################################
# 格子を出力
# 引数には1次元配列を渡す(Fortran配列の順番)
###############################################################################
iric.cg_iRIC_Write_Grid3d_Coords(fid, isize, jsize, ksize, grid_x_arr_3d.flatten(order='F'), grid_y_arr_3d.flatten(order='F'), grid_z_arr_3d.flatten(order='F'))

###############################################################################
# メモリ確保
###############################################################################

# 計算結果を格納する配列を初期化
# インデックス番号をそのまま値とする
# ノード用 [isize, jsize, ksize]
node_i, node_j, node_k = np.indices((isize, jsize, ksize))

# セル用 [isize-1, jsize-1, ksize-1]
cell_i, cell_j, cell_k = np.indices((isize-1, jsize-1, ksize-1))

# i-face用 [isize, jsize-1, ksize]
iface_i, iface_j, iface_k = np.indices((isize, jsize-1, ksize))

# j-face用 [isize-1, jsize, ksize]
jface_i, jface_j, jface_k = np.indices((isize-1, jsize, ksize))

# k-face用 [isize, jsize, ksize-1]
kface_i, kface_j, kface_k = np.indices((isize, jsize, ksize - 1))

# ベクトルはx方向はインデックスi/isize、y方向はインデックスj/jsize、z方向はインデックスk/ksize
result_vector_x = np.zeros((isize, jsize, ksize), dtype=np.float64)
result_vector_y = np.zeros((isize, jsize, ksize), dtype=np.float64)
result_vector_z = np.zeros((isize, jsize, ksize), dtype=np.float64)

for i in range(isize):
    result_vector_x[i, :, :] = i/isize
for j in range(jsize):
    result_vector_y[:, j, :] = j/jsize
for k in range(ksize):
    result_vector_z[:, :, k] = k/ksize

i_tmp = 0
j_tmp = 0
k_tmp = 0

###############################################################################
# 計算結果を計算
###############################################################################
for t in range(0, time_end+1):

    ###########################################################################
    # パーティクルの位置の計算、パーティクルの座標はgrid_x_arr_3d(i_tmp, j_tmp, k_tmp)、grid_y_arr_3d(i_tmp, j_tmp, k_tmp)、grid_z_arr_3d(i_tmp, j_tmp, k_tmp)で与えられる。
    # インデックスi_tmp, j_tmp, k_tmpは0~isize-1, 0~jsize-1, 0~ksize-1の範囲で変化する。
    # 移動のルールは以下の通り
    # パーティクルは(0, 0, 0)からスタートし、格子の外周を時計回りに1ステップずつ移動する。
    # 格子の外周に到達したら、z方向に1ステップ移動し、再度格子の外周を時計回りに1ステップずつ移動する。
    # z方向にksize-1ステップ移動したら、k_tmp=0に戻り、再度格子の外周を時計回りに1ステップずつ移動する。
    if t > 0:
        # 格子の外周を時計回りに1ステップずつ移動
        if j_tmp == 0 and i_tmp < isize-1:
            i_tmp += 1
        elif i_tmp == isize-1 and j_tmp < jsize-1:
            j_tmp += 1
        elif j_tmp == jsize-1 and i_tmp > 0:
            i_tmp -= 1
        elif i_tmp == 0 and j_tmp > 0:
            j_tmp -= 1

        # 外周を1周したらz方向に1進む
        if i_tmp == 0 and j_tmp == 0:
            k_tmp += 1
            if k_tmp == ksize:
                k_tmp = 0

    # 各種値の出力
    iric.cg_iRIC_Write_Sol_Start(fid)
    iric.cg_iRIC_Write_Sol_Time(fid, float(t))
    iric.cg_iRIC_Write_Sol_Node_Real(fid,"vectorX", result_vector_x.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_Node_Real(fid,"vectorY", result_vector_y.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_Node_Real(fid,"vectorZ", result_vector_z.flatten(order='F'))

    # ノード i,j,k
    iric.cg_iRIC_Write_Sol_Node_Integer(fid, "node_index_i", node_i.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_Node_Integer(fid, "node_index_j", node_j.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_Node_Integer(fid, "node_index_k", node_k.flatten(order='F'))

    # セル i,j,k
    iric.cg_iRIC_Write_Sol_Cell_Integer(fid, "cell_index_i", cell_i.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_Cell_Integer(fid, "cell_index_j", cell_j.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_Cell_Integer(fid, "cell_index_k", cell_k.flatten(order='F'))

    # i-face i,j,k
    iric.cg_iRIC_Write_Sol_IFace_Integer(fid, "iface_index_i", iface_i.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_IFace_Integer(fid, "iface_index_j", iface_j.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_IFace_Integer(fid, "iface_index_k", iface_k.flatten(order='F'))

    # j-face i,j,k
    iric.cg_iRIC_Write_Sol_JFace_Integer(fid, "jface_index_i", jface_i.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_JFace_Integer(fid, "jface_index_j", jface_j.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_JFace_Integer(fid, "jface_index_k", jface_k.flatten(order='F'))

    # k-face i,j,k
    iric.cg_iRIC_Write_Sol_KFace_Integer(fid, "kface_index_i", kface_i.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_KFace_Integer(fid, "kface_index_j", kface_j.flatten(order='F'))
    iric.cg_iRIC_Write_Sol_KFace_Integer(fid, "kface_index_k", kface_k.flatten(order='F'))


    # パーティクルの位置の出力
    iric.cg_iRIC_Write_Sol_ParticleGroup_GroupBegin(fid, "particle")
    iric.cg_iRIC_Write_Sol_ParticleGroup_Pos3d(fid, grid_x_arr_3d[i_tmp, j_tmp, k_tmp], grid_y_arr_3d[i_tmp, j_tmp, k_tmp], grid_z_arr_3d[i_tmp, j_tmp, k_tmp])
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vectorX", result_vector_x[i_tmp, j_tmp, k_tmp])
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vectorY", result_vector_y[i_tmp, j_tmp, k_tmp])
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vectorZ", result_vector_z[i_tmp, j_tmp, k_tmp])
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vector_xX", result_vector_x[i_tmp, j_tmp, k_tmp])
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vector_xY", 0.0)
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vector_xZ", 0.0)
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vector_yX", 0.0)
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vector_yY", result_vector_y[i_tmp, j_tmp, k_tmp])
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vector_yZ", 0.0)
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vector_zX", 0.0)
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vector_zY", 0.0)
    iric.cg_iRIC_Write_Sol_ParticleGroup_Real(fid, "particle_vector_zZ", result_vector_z[i_tmp, j_tmp, k_tmp])
    iric.cg_iRIC_Write_Sol_ParticleGroup_GroupEnd(fid)

    # コンソールに計算進捗を出力
    print("Time: " + str(t) + " / " + str(time_end))


iric.cg_iRIC_Close(fid)