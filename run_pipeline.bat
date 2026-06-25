@echo off
echo ============================================================
echo Motion-Blindness Pipeline Runner
echo ============================================================
echo.

call venv\Scripts\activate

echo [Stage 1] Generating dummy data...
python scripts/stage1_generate_dummy_data.py
if %errorlevel% neq 0 goto error

echo.
echo [Stage 2] Verifying depth ground truth...
python scripts/stage2_verify_depth_groundtruth.py
if %errorlevel% neq 0 goto error

echo.
echo [Stage 3] Building annotation spreadsheet...
python scripts/stage3_build_annotation_spreadsheet.py
if %errorlevel% neq 0 goto error

echo.
echo [Stage 6] Compiling results (no model outputs yet)...
python scripts/stage6_compile_results.py

echo.
echo [Stage 7] Generating visualizations...
python scripts/stage7_visualize.py
if %errorlevel% neq 0 goto error

echo.
echo ============================================================
echo Pipeline complete. Check results/ and paper/figures/
echo ============================================================
goto end

:error
echo.
echo [ERROR] Pipeline stopped at error above.
exit /b 1

:end
