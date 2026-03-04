# SUMO 安装与路径检查

## 1. 检查是否已安装且路径正确

在项目根目录运行：

```
python check_sumo.py
```

会检查：
- `SUMO_HOME` 是否设置、指向的目录是否存在
- `sumo`、`sumo-gui` 是否在 `SUMO_HOME\bin` 或系统 PATH 中
- 运行 `sumo --version` 是否正常

---

## 2. 你电脑上的安装位置（根据检查结果）

常见安装路径示例：
- `C:\Program Files (x86)\Eclipse\Sumo`
- `C:\Program Files\Eclipse\Sumo`

`sumo.exe`、`sumo-gui.exe` 应在其中的 `bin` 子目录下，例如：
- `C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe`
- `C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe`

---

## 3. 设置/修改 SUMO 路径，让 Cursor 能用

### 方法 A：系统环境变量（推荐，一次设置永久有效）

1. 按 `Win + R`，输入 `sysdm.cpl`，回车  
2. 打开 **高级** → **环境变量**  
3. 在 **用户变量** 或 **系统变量** 中：
   - 若无 `SUMO_HOME`：点 **新建**
   - 变量名：`SUMO_HOME`  
   - 变量值：SUMO 安装的**根目录**，例如  
     `C:\Program Files (x86)\Eclipse\Sumo`  
     （不要写 `\bin`，只到 `Sumo` 这一层）
4. 确定保存后，**完全关掉 Cursor 再重新打开**，新开一个终端再运行 `python check_sumo.py` 或 `main.py`。

（可选）把 `bin` 加入 PATH，便于在终端直接打 `sumo`、`sumo-gui`：
- 编辑 **Path**，新增一行：`%SUMO_HOME%\bin`

---

### 方法 B：在 Cursor 终端里临时设置（只对当前终端有效）

在 **PowerShell** 中，每次新开终端后执行一次：

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

把路径换成你自己 `check_sumo.py` 里看到的安装目录。  
然后再运行：

```powershell
python main.py -c brussels_rural_config.json -m simulate
```

或带 GUI：

```powershell
python main.py -c brussels_rural_config.json -m simulate --gui
```

---

### 方法 C：如果 Cursor 里仍拿不到 SUMO_HOME

`main.py` 在 **Windows** 下会尝试这些默认路径（当 `SUMO_HOME` 未设置时）：

- `C:\Program Files (x86)\Eclipse\Sumo`
- `C:\Program Files\Eclipse\Sumo`

只要你的 SUMO 安装在这些之一（且下面有 `bin\sumo.exe`），即使 Cursor 里没有 `SUMO_HOME`，脚本也会尝试用它们。

---

## 4. 用 Conda 环境时

若使用项目的 conda 环境 `crowdsourced_road_damage_estimation`，先激活再跑检查和 main：

```powershell
conda activate crowdsourced_road_damage_estimation
python check_sumo.py
python main.py -c brussels_rural_config.json -m simulate
```

`SUMO_HOME` 和 PATH 的规则与上面相同；环境变量是在“系统/用户”或“当前终端”里设的，Conda 会继承。

---

## 5. 无反应、报错时

1. 再跑一次：`python check_sumo.py`，确认输出里 `sumo`、`sumo-gui` 都是 “OK” 或 “reachable”。  
2. 若 `SUMO_HOME` 为 “NOT SET”：
   - 按 **方法 A** 或 **方法 B** 设置并（若用 A）重启 Cursor。  
3. 若 `sumo --version` 失败：
   - 检查安装目录下是否有 `bin\sumo.exe`；  
   - 若装在其他盘或目录，把 `SUMO_HOME` 设为**那个**根目录。  
4. 若 `main.py` 报 `ModuleNotFoundError`（如 `shapely`）：
   - 先创建并激活 conda 环境：  
     `conda env create -n crowdsourced_road_damage_estimation -f env/environment.yml`  
     `conda activate crowdsourced_road_damage_estimation`
