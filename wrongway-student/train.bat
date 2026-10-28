@echo off
title Train Bot 25.py
echo ========================================================
echo   BAT DAU HUAN LUYEN BOT 25.PY (GENETIC ALGORITHM)
echo ========================================================
echo Cac tham so mac dinh:
echo   - So the he (generations): 5
echo   - Quy mo quan the (pop-size): 6
echo   - So luong CPU song song: 6
echo   - Doi thu: hihi.py, t1.py (Champion), minimax_alpha_beta_pruning.py
echo.
python train.py --bot 25.py --generations 5 --pop-size 6 --workers 6 --apply
pause

