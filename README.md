### Отчёт по сравнению производительности и качества работы алгоритмов YOLOv8 и Detectron2 для распознавания людей на видео

---

#### **1. Введение**

Цель данного отчета — сравнить два алгоритма распознавания объектов, YOLOv8 и Detectron2, применительно к задаче детекции людей на видеофайле `crowd.mp4`. Рассмотрим производительность и качество детекции каждого алгоритма, выберем наиболее предпочтительный и предложим возможные шаги по улучшению качества и производительности.

---

#### **2. Сравнение производительности и качества работы алгоритмов**

| Параметр                  | YOLOv8                    | Detectron2                |
|---------------------------|---------------------------|---------------------------|
| **Оптимальный threshold** | 0.3           | 0.55     |
| **Точность распознавания объктов вблизи** | Хорошая (70-80%)           | Очень хорошая (85-95%)     |
| **Точность распознавания объктов вдалеке** | Низкая (0-10%)           | Хорошая (70-80%)     |
| **Уверенность детекции**   | Умеренная (выше 0.5)       | Высокая (выше 0.7)         |
| **Количество ложных срабатываний** | Низкое                 | Низкое                    |
| **Среднее время инференса**| ~25 мс на кадр (~40 FPS)   | ~1 с на кадр (~1 FPS)  |
| **Скорость обработки**     | Высокая (подходит для реального времени) | Низкая (подходит для оффлайн обработки) |
| **Вес модели**             | Легкая (20-50 МБ)          | Тяжелая (200-500 МБ)       |
| **Требования к ресурсам**  | Низкие (можно использовать CPU) | Высокие (лучше использовать GPU) |

**Анализ:**

- **YOLOv8** демонстрирует быструю работу, обеспечивая высокую частоту кадров (до 40 FPS), что делает его подходящим для приложений, требующих реальной времени. Однако, качество распознавания у него несколько уступает Detectron2, особенно на сложных кадрах с перекрытием объектов.
- **Detectron2** более точен в распознавании людей, особенно в сложных сценах, однако, его производительность ниже — около 1 FPS. Этот алгоритм лучше справляется с задачами, где критически важна точность, а не скорость. Для эффективной работы требует использования GPU.

---

#### **3. Обоснование выбора наиболее предпочтительного алгоритма**

На основе сравнения можно сделать следующие выводы:

- Если цель состоит в **реализации детекции в реальном времени**, особенно в условиях с ограниченными вычислительными ресурсами, **YOLOv8** является более предпочтительным выбором. Он обеспечивает высокую скорость инференса с приемлемым уровнем точности, что делает его идеальным для видеонаблюдения, потоковых приложений или мобильных устройств.
  
- **Detectron2** следует выбрать для задач, где требуется **максимальная точность распознавания**, например, в медицинских или промышленных приложениях, где скорость не так критична, но каждая ошибка может иметь значительные последствия. Detectron2 лучше справляется с детекцией на сложных кадрах с множеством объектов, но требует больше ресурсов и подходит для обработки в оффлайне.

### Вывод:
- Для задач **реального времени** — предпочтителен **YOLOv8**.
- Для задач, где важна **высокая точность**, например, анализ видео в оффлайн-режиме — **Detectron2**.

---

#### **4. Шаги по дальнейшему улучшению качества распознавания и производительности**

Для улучшения производительности и точности работы алгоритмов можно предпринять следующие шаги:

1. **Оптимизация моделей для специфической задачи**:
   - Провести дообучение (fine-tuning) обеих моделей на доменном наборе данных с людьми, чтобы повысить точность на конкретных данных.
   - Уменьшить размер моделей для повышения скорости обработки на устройствах с ограниченными ресурсами.

2. **Улучшение инференса на GPU**:
   - Для Detectron2 можно использовать оптимизированные библиотеки, такие как TensorRT, что позволит увеличить скорость инференса на GPU.

3. **Аугментации данных**:
   - Использование агрессивных аугментаций данных, таких как увеличение яркости, повороты или увеличение контраста, поможет сделать модели более устойчивыми к различным условиям съёмки.

4. **Использование механизма NMS (non-maximum suppression)**:
   - Улучшение постобработки, например, использование продвинутого метода подавления перекрывающихся объектов (NMS), позволит уменьшить количество ложных срабатываний и повысить точность.

5. **Использование гибридных подходов**:
   - Возможна реализация гибридного решения, где YOLOv8 используется для детекции в реальном времени, а более точный Detectron2 применялся бы для последующей оффлайн-анализа сложных сцен.

6. **Распараллеливание инференса**:
   - Для больших видеофайлов можно распараллелить инференс, распределяя обработку разных частей видео на несколько GPU или процессоров.

---

### **Заключение**

В зависимости от конкретных требований (точность или скорость) можно выбрать YOLOv8 или Detectron2. Для дальнейшего повышения производительности и качества детекции можно рассмотреть методы оптимизации, дообучения и распараллеливания, что позволит более эффективно решать задачи детекции людей в различных условиях.

--- 

**Подготовил:** Данилова Светлана

**Дата:** 17.10.2024
