# RoseLoRA
Реализация алгоритма [RoseLoRA][1]. Метод реализован в качестве обертки над алгоритмом LoRA из бибилиотеки PEFT.

## Данные
В данном проекте анализируются 3 набора данных: `MRPC` из датасета `GLUE` для задачи бинарной классификации текстов, `AqUA` для задачи выбора правильного ответа и `ZSRE` для Knowledge Editing. 

## Модели
- На задаче `MRPC` и `AqUA` проводится сравнение LoRA и RoseLoRA для архитектуры `bert-base-uncased` из библиотеки transformers.
- Для Knowledge Editing использована модель `GPT-2`, обучение и инференс проводились в библиотеке [EasyEdit][2]

## Использование
Реализация алгоритма `RoseLoRA` и описание доступных аргументов находятся в `src\rose_lora.py`. Запуск метода на основе библиотек HuggingFace осуществляется всего двумя строками кода.
Создание модели:
```
base_model = ... #любая модель из transformers
peft_config = LoraConfig(...) #конфигурация LoRA из peft
roselora_model = RoseLoraModel(base_model, peft_config, total_step = max_epochs * iter_in_epoch)
```
Тренировка (необходим любой кастомный trainer):
```
...
opt.step()
self.model.update_and_allocate(i_step) #здесь i_step - номер итерации
...
```


[1]:https://arxiv.org/abs/2406.10777
[2]:https://github.com/zjunlp/EasyEdit
