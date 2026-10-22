# Проверка workflow релиза — 22 октября 2026

Обновление: текущий workflow публикует непосредственно `MeanVC2Desktop.exe`.
Упаковка в ZIP удалена, README и текст релиза обновлены. Actionlint проходит.
Локально выполнены проверка размера настоящего EXE, шесть граничных случаев
валидации и публикация с подставным `gh`: загружается EXE, ошибка загрузки
останавливает публикацию. Реальный запуск GitHub Actions не выполнялся.

Ниже сохранены результаты первоначальной проверки варианта с ZIP.

Workflow: `.github/workflows/release.yml`, ручной запуск `workflow_dispatch`.
Один Windows runner собирает CUDA-enabled onefile EXE с поддержкой CPU,
проверяет его и публикует единственный ZIP через GitHub CLI.

Проверено локально:

* `actionlint 1.7.12`: ошибок нет. PowerShell-шаги разобраны без синтаксических ошибок.
* 14 автоматических тестов прошли, включая оба варианта PyInstaller: onefile и onedir.
* Проверены запрет существующего/некорректного тега и остановка публикации при ошибке загрузки.
* Реально собран `dist/cuda-onefile/MeanVC2Desktop.exe`: 2 036 876 291 байт.
* Точный шаг упаковки из workflow создал `dist/MeanVC2Desktop-Windows-x64.zip`:
  2 036 876 425 байт, меньше 2 ГиБ. Внутри ровно `MeanVC2Desktop.exe`; CRC проверен.
* Этот EXE успешно выполнил по два прохода реального streaming-инференса на CPU
  и CUDA:0 (RTX 4060 Ti), по 2,08 с на проход, выход конечный и ненулевой.
  Запуск выполнялся из `%TEMP%`, с PATH только из системных Windows-каталогов,
  пустыми PYTHONHOME/PYTHONPATH и офлайн-кешем моделей.
* Диагностика EXE и offscreen-запуск GUI прошли. Для шрифтов в offscreen-режиме
  workflow задаёт `QT_QPA_FONTDIR` из системной папки Windows Fonts.

Полная onefile-сборка со всеми библиотеками превышала лимит GitHub: 2 427 461 994 байта.
Onefile исключает только `cudnn_engines_precompiled64_9.dll`, поскольку исходный
инференс уже задаёт `torch.backends.cudnn.enabled = False`. CPU/CUDA-инференс
проверен после этого исключения. Обычная onedir-сборка библиотеку сохраняет.
При включении cuDNN её нужно вернуть.

Детали: `release-package.json`, `release-workflow-tests.txt`, `onefile-cpu.json`,
`onefile-cuda.json`, `onefile-gui.png`. CUDA Toolkit и внешний Python в PATH
для проверок не использовались; совместимый драйвер NVIDIA установлен.

Сам workflow на GitHub не запускался: у локального репозитория не настроен remote.
На стандартном GitHub runner без GPU workflow проверяет тесты, импорт CUDA-enabled
runtime и GUI; физическое аудио и CUDA-инференс требуют отдельного оборудования.
