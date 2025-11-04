@echo off
REM 批量评估所有RL检查点 (Windows版本)

REM 配置
set CHECKPOINT_DIR=checkpoints\rl_model
set OUTPUT_DIR=evaluation_results
set EVAL_SAMPLES=100

REM 创建输出目录
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM 评估所有检查点
echo 开始批量评估检查点...
echo 检查点目录: %CHECKPOINT_DIR%
echo 输出目录: %OUTPUT_DIR%
echo 评估样本数: %EVAL_SAMPLES%
echo.

REM 遍历所有checkpoint目录
for /d %%d in (%CHECKPOINT_DIR%\checkpoint-*) do (
    if exist "%%d" (
        echo ==========================================
        echo 评估检查点: %%~nxd
        echo ==========================================
        
        set output_file=%OUTPUT_DIR%\evaluation_results_%%~nxd.json
        
        REM 执行评估
        python evaluation\evaluate_checkpoint.py ^
            --checkpoint_path "%%d" ^
            --eval_samples %EVAL_SAMPLES% ^
            --output_file "%output_file%"
        
        if %errorlevel% equ 0 (
            echo [成功] %%~nxd 评估完成
        ) else (
            echo [失败] %%~nxd 评估失败
        )
        echo.
    )
)

echo ==========================================
echo 批量评估完成！
echo 结果保存在: %OUTPUT_DIR%
echo ==========================================
pause

