# Переход на официальный репозиторий MeanVC2 — 22 октября 2026

MeanVC2 подключён как git submodule `third_party/MeanVC2` на commit
`13acf84c1bf135ea5edad9c245b345289b06b33e`. `git diff` и `git status --short`
в submodule не показывают изменений исходников. Локальные `__pycache__`
исключены через Git info/exclude; это не меняет файлы upstream.

Копия `app/vendor/meanvc2` удалена. `app/upstream.py` импортирует официальный
`runtime/run_rt.py` и `runtime/src` напрямую. Адаптер отвечает за пути кеша,
валидацию весов, освобождение speaker encoder после получения embedding и
контекст выбранной CUDA-карты. Потоковые ASR/BN, DiT-кэши и overlap-add
вокодера используют методы upstream. При двух шагах ODE вызывается оригинальный
метод; другое число шагов реализовано отдельным расширением адаптера.
Оставшийся минимальный S3PRL проверен по исходным SHA256 с нормализацией CRLF.

В `build.ps1` добавлена инициализация зафиксированного submodule; checkout
в workflow получает submodules рекурсивно. В EXE включены шесть Python-модулей
официального runtime и две JSON-конфигурации, без исходников обучения.
Состав проверен через Analysis TOC и чтение архивов внутри готового EXE.

Проверки выполнены на Windows 11, i5-12400F, RTX 4060 Ti, Python 3.11.9,
PyTorch 2.5.1 CPU / CUDA 12.1, с реальными кешированными весами MeanVC2-ru.

| Проверка | Результат |
|---|---|
| Unit/integration tests | Все 18 прошли; проверяются происхождение runtime, вызов штатного ODE, кэши и rollover расширения, выбор GPU-контекста, упаковка, прежние проверки аудио/моделей/GUI |
| CPU-only PyTorch | Два прохода по 10,08 с; конечный ненулевой выход, сброс кешей, два открытия реальных устройств; [JSON](submodule-cpu.json) |
| CUDA PyTorch | Два прохода по 10,08 с на RTX 4060 Ti; [JSON](submodule-cuda.json) |
| Настоящий GUI / Windows audio | По 65 блоков CPU и CUDA, смена физических/виртуальных устройств, запись через виртуальный вход, прослушивание в кабель, восстановление настроек; [JSON](submodule-gui-integration.json) |
| Onefile EXE, CPU Auto | CUDA скрыта через `CUDA_VISIBLE_DEVICES=-1`, выбран CPU; два прохода с реальными весами; [JSON](submodule-onefile-auto.json) |
| Тот же EXE, CUDA | Выбран cuda:0, два прохода с реальными весами; [JSON](submodule-onefile-cuda.json) |
| Самостоятельность EXE | Benchmark и GUI запущены из TEMP с PATH только Windows/System32, без PYTHONHOME/PYTHONPATH; модели из пользовательского кеша |
| GUI EXE без доступной CUDA | Успешный запуск и [снимок окна](submodule-onefile-gui.png) |
| GitHub workflow | actionlint проходит; реальный запуск Actions не выполнен: remote не настроен |
| ZIP | Ровно `MeanVC2Desktop.exe`, CRC исправен, размер 2 036 875 257 байт (<2 GiB); [состав и SHA256](submodule-package.json) |

В готовом EXE среднее время блока 160 мс: CPU Auto — 42,3–42,4 мс,
CUDA — 35,3–39,2 мс. Это время инференса, не полная задержка тракта.
Замеры сделаны на рабочем компьютере, с обычными фоновыми приложениями.
В GUI оценка задержки составила примерно 559 мс CPU / 683 мс CUDA;
физический loopback не измерялся. В CPU-сценарии GUI зарегистрирован один
output underflow, в CUDA — ноль; входных переполнений не было.

При тестировании реального аудио Qt offscreen дал PortAudio -9999 на старте
WASAPI. Короткая проверка воспроизвела зависимость от Qt backend; тот же
сценарий со штатным Windows backend прошёл полностью. В Actions offscreen
используется только для запуска окна, без открытия аудиопотоков.
Физический микрофон при записи дал тишину, образец корректно отклонён;
запись речи через виртуальный вход прошла. Пользовательские голоса и настройки
не менялись: GUI-сценарии выполнялись в отдельной директории данных с общим
кешем моделей.

Не проверены в этом изменении: отдельный компьютер без Python, Windows 10,
вторая физическая GPU, субъективное качество звука и физическая задержка.
Скрытие GPU и очищенный PATH проверяют fallback и комплектность библиотек,
но не заменяют испытание на чистом компьютере.

Артефакты:

- `dist/MeanVC2Desktop-Windows-x64.zip` — один EXE с CPU и CUDA;
- `dist/cuda-onefile/MeanVC2Desktop.exe` — тот же исполняемый файл;
- воспроизведение: `powershell -ExecutionPolicy Bypass -File scripts/build.ps1 -Variant cuda -OneFile`.
