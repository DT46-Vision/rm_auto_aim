# 定义一个消息结构，用于描述目标对象的状态和属性

std_msgs/Header header  # 消息的头部信息，通常包含时间戳和帧ID等

bool tracking           # 表示目标是否正在被跟踪
string id               # 目标的唯一标识符
int32 armors_num        # 目标身上装甲的数量

geometry_msgs/Point position    # 目标的位置信息
geometry_msgs/Vector3 velocity  # 目标的线速度信息

float64 yaw            # 目标的偏航角（围绕Z轴的旋转角度）
float64 v_yaw          # 目标的角速度（偏航方向上的旋转速度）
float64 radius_1       # 目标的第一个半径参数，用于描述目标的形状或尺寸
float64 radius_2       # 目标的第二个半径参数，用于描述目标的形状或尺寸
float64 dz             # 目标在Z方向上的尺寸或高度

