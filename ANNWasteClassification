{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN6FLQrLj0id5reY/ddE2UB"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9u50O-N3iwbJ"
      },
      "outputs": [],
      "source": [
        "#IMPORTS FOR MODEL\n",
        "import numpy as np\n",
        "import os\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, InputLayer\n",
        "from tensorflow.keras.preprocessing import image\n",
        "from matplotlib import pyplot as plt\n",
        "\n",
        "from google.colab import files\n",
        "import zipfile\n",
        "import os\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "#Set Preproccess image size and Load/Unzip Dataset into its directories\n",
        "img_height, img_width = 64, 64\n",
        "\n",
        "with zipfile.ZipFile('Dataset.zip', 'r') as zip_ref:\n",
        "  zip_ref.extractall('images/')\n",
        "\n",
        "compost_dir = 'images/Dataset/Compost'\n",
        "general_dir = 'images/Dataset/General'\n",
        "recycle_dir = 'images/Dataset/Recycle'\n"
      ],
      "metadata": {
        "id": "tT_7nej5VtZZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Preprocess Data and each of their titles to a different number\n",
        "def load_images(folder, label):\n",
        "  imgs = []\n",
        "  labels = []\n",
        "  for filename in os.listdir(folder):\n",
        "    if filename.endswith('.jpg'):\n",
        "        img_path = os.path.join(folder, filename)\n",
        "        img = image.load_img(img_path, target_size=(img_height, img_width))\n",
        "        img_array = image.img_to_array(img)\n",
        "        img_array = img_array.flatten() / 255.0\n",
        "        imgs.append(img_array)\n",
        "        labels.append(label)\n",
        "  return np.array(imgs), np.array(labels)\n",
        "\n",
        "compost_imgs, compost_labels = load_images(compost_dir, 0)\n",
        "general_imgs, general_labels = load_images(general_dir, 1)\n",
        "recycle_imgs, recycle_labels = load_images(recycle_dir, 2)\n"
      ],
      "metadata": {
        "id": "ONHN3A6rXAnZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Combine images data\n",
        "X_train = np.vstack((compost_imgs, general_imgs, recycle_imgs))\n",
        "\n",
        "# Combine labels\n",
        "y_train = np.hstack((compost_labels, general_labels, recycle_labels))\n"
      ],
      "metadata": {
        "id": "3V4ovUhSZdU0"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Creating ANN and Defining Layers\n",
        "model = Sequential()\n",
        "model.add(InputLayer(input_shape=(X_train.shape[1],)))    # input layer\n",
        "model.add(Dense(64, activation='relu'))                    # hidden layer\n",
        "model.add(Dense(3, activation='softmax'))                  # output layer\n",
        "\n",
        "model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
        "print(model.summary())"
      ],
      "metadata": {
        "id": "ZlwVPzXsZfEw",
        "collapsed": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 251
        },
        "outputId": "4b8265b9-e592-42f3-996a-fb4fc3337de1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/input_layer.py:27: UserWarning: Argument `input_shape` is deprecated. Use `shape` instead.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1mModel: \"sequential\"\u001b[0m\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ dense (\u001b[38;5;33mDense\u001b[0m)                   │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m64\u001b[0m)             │       \u001b[38;5;34m786,496\u001b[0m │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                 │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m3\u001b[0m)              │           \u001b[38;5;34m195\u001b[0m │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
              "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
              "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
              "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                   │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)             │       <span style=\"color: #00af00; text-decoration-color: #00af00\">786,496</span> │\n",
              "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
              "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">3</span>)              │           <span style=\"color: #00af00; text-decoration-color: #00af00\">195</span> │\n",
              "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m786,691\u001b[0m (3.00 MB)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">786,691</span> (3.00 MB)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m786,691\u001b[0m (3.00 MB)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">786,691</span> (3.00 MB)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
            ],
            "text/html": [
              "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
              "</pre>\n"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "None\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.utils import to_categorical\n",
        "\n",
        "# Convert labels to catagorical waste formats\n",
        "y_train_cat = to_categorical(y_train)\n",
        "\n",
        "# Train model\n",
        "history = model.fit(X_train, y_train_cat, epochs=50, verbose=1)\n"
      ],
      "metadata": {
        "id": "3ADoG3vSaUsh",
        "collapsed": true,
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d75ce2c0-65d8-40ef-ef4b-baf4d26ab5df"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 21ms/step - accuracy: 0.3085 - loss: 6.7856 \n",
            "Epoch 2/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.2514 - loss: 5.1141\n",
            "Epoch 3/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 20ms/step - accuracy: 0.3587 - loss: 3.8274\n",
            "Epoch 4/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 25ms/step - accuracy: 0.2773 - loss: 1.6387\n",
            "Epoch 5/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.4410 - loss: 1.3062\n",
            "Epoch 6/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 20ms/step - accuracy: 0.4394 - loss: 1.0763\n",
            "Epoch 7/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.5739 - loss: 1.2449\n",
            "Epoch 8/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 20ms/step - accuracy: 0.5246 - loss: 0.9855\n",
            "Epoch 9/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.4957 - loss: 1.1637\n",
            "Epoch 10/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.5200 - loss: 1.0188\n",
            "Epoch 11/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.6202 - loss: 0.9480\n",
            "Epoch 12/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 20ms/step - accuracy: 0.4527 - loss: 1.1354\n",
            "Epoch 13/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 21ms/step - accuracy: 0.6711 - loss: 0.7409\n",
            "Epoch 14/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 20ms/step - accuracy: 0.5492 - loss: 1.0734\n",
            "Epoch 15/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.6187 - loss: 0.8018\n",
            "Epoch 16/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 20ms/step - accuracy: 0.6617 - loss: 0.7304\n",
            "Epoch 17/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.6218 - loss: 0.7979\n",
            "Epoch 18/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 24ms/step - accuracy: 0.5787 - loss: 0.8275\n",
            "Epoch 19/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.7274 - loss: 0.7076\n",
            "Epoch 20/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.6569 - loss: 0.6850 \n",
            "Epoch 21/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.7290 - loss: 0.7024\n",
            "Epoch 22/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.7666 - loss: 0.5830\n",
            "Epoch 23/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 17ms/step - accuracy: 0.6616 - loss: 0.6900\n",
            "Epoch 24/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.7384 - loss: 0.6275\n",
            "Epoch 25/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.7188 - loss: 0.6021\n",
            "Epoch 26/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.7236 - loss: 0.6587\n",
            "Epoch 27/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.6491 - loss: 0.6557\n",
            "Epoch 28/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.7697 - loss: 0.6232\n",
            "Epoch 29/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.6445 - loss: 0.8091\n",
            "Epoch 30/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.8269 - loss: 0.5549\n",
            "Epoch 31/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.7463 - loss: 0.5483\n",
            "Epoch 32/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 23ms/step - accuracy: 0.7885 - loss: 0.5055\n",
            "Epoch 33/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.7635 - loss: 0.4695\n",
            "Epoch 34/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.8771 - loss: 0.4129\n",
            "Epoch 35/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.8426 - loss: 0.4286\n",
            "Epoch 36/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.8581 - loss: 0.4249\n",
            "Epoch 37/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.8386 - loss: 0.4281\n",
            "Epoch 38/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.8535 - loss: 0.4379\n",
            "Epoch 39/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step - accuracy: 0.8324 - loss: 0.4310\n",
            "Epoch 40/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 20ms/step - accuracy: 0.7878 - loss: 0.4952\n",
            "Epoch 41/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 20ms/step - accuracy: 0.9053 - loss: 0.3869\n",
            "Epoch 42/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.8489 - loss: 0.3893\n",
            "Epoch 43/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 20ms/step - accuracy: 0.8191 - loss: 0.4415\n",
            "Epoch 44/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 25ms/step - accuracy: 0.8551 - loss: 0.3810\n",
            "Epoch 45/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.8489 - loss: 0.4045\n",
            "Epoch 46/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 27ms/step - accuracy: 0.8121 - loss: 0.4288\n",
            "Epoch 47/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.7823 - loss: 0.4774\n",
            "Epoch 48/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 20ms/step - accuracy: 0.8904 - loss: 0.3601\n",
            "Epoch 49/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step - accuracy: 0.8763 - loss: 0.3564\n",
            "Epoch 50/50\n",
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 20ms/step - accuracy: 0.9287 - loss: 0.3298\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#RESULTS FOR MODEL AFTER TRAINED\n",
        "\n",
        "#Overall Model Eval\n",
        "model.evaluate(X_train, y_train_cat)\n",
        "\n",
        "#Accuracy Plot\n",
        "plt.plot(history.history['accuracy'])\n",
        "plt.title('Model Accuracy')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Accuracy')\n",
        "plt.grid(True)\n",
        "plt.show()\n",
        "\n",
        "#Loss Plot\n",
        "plt.plot(history.history['loss'])\n",
        "plt.title('Model Loss')\n",
        "plt.xlabel('Epoch')\n",
        "plt.ylabel('Loss')\n",
        "plt.grid(True)\n",
        "plt.show()\n"
      ],
      "metadata": {
        "id": "MnO9blAT3tWr",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 944
        },
        "outputId": "1164ad68-0ad8-4d0a-a3c2-371817e0583f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m3/3\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 13ms/step - accuracy: 0.9537 - loss: 0.2847 \n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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00kqlQojR45EWlqa+TGj0Yi0tDSMHTvWhSPzDL169UJUVJTF/FdUVGDPnj2c/w4QRRELFy7Ed999h82bN6NXr14Wz48cORJKpdJivk+fPo3s7GzOt50YjUZotVrOtQNMnDgRx44dw+HDh82/Ro0ahTlz5ph/zzl3nKqqKpw9exbR0dGO//7udEoyiatXrxbVarX4ySefiCdPnhQfeughMSgoSMzPz3f10NxCZWWleOjQIfHQoUMiAHH58uXioUOHxKysLFEURfGVV14Rg4KCxB9++EE8evSoeNNNN4m9evUSa2pqXDzy7ueRRx4RAwMDxS1btoh5eXnmX9XV1eZrHn74YbFnz57i5s2bxf3794tjx44Vx44d68JRd19PP/20uHXrVvH8+fPi0aNHxaeffloUBEHctGmTKIqca2dofFpKFDnn9vTUU0+JW7ZsEc+fPy/+/vvv4qRJk8SwsDCxsLBQFEXHzjWDGzv597//Lfbs2VNUqVTimDFjxN27d7t6SG7jt99+EwE0+3XfffeJomg6Dv7888+LkZGRolqtFidOnCiePn3atYPuplqaZwDixx9/bL6mpqZGfPTRR8Xg4GDRx8dHvOWWW8S8vDzXDbobu//++8X4+HhRpVKJ4eHh4sSJE82BjShyrp2haXDDObefWbNmidHR0aJKpRJjY2PFWbNmiZmZmebnHTnXgiiKYufXf4iIiIi6BubcEBERkVthcENERERuhcENERERuRUGN0RERORWGNwQERGRW2FwQ0RERG6FwQ0RERG5FQY3ROTxBEHA999/7+phEJGdMLghIpf64x//CEEQmv2aNm2aq4dGRN2UwtUDICKaNm0aPv74Y4vH1Gq1i0ZDRN0dV26IyOXUajWioqIsfgUHBwMwbRm99957mD59Ory9vZGYmIivv/7a4vXHjh3D9ddfD29vb4SGhuKhhx5CVVWVxTUrV67EoEGDoFarER0djYULF1o8X1xcjFtuuQU+Pj7o27cvfvzxR8d+aCJyGAY3RNTlPf/887jttttw5MgRzJkzB7Nnz8apU6cAABqNBlOnTkVwcDD27duHtWvX4tdff7UIXt577z0sWLAADz30EI4dO4Yff/wRffr0sXiPpUuX4s4778TRo0cxY8YMzJkzB6WlpU79nERkJ3Zpv0lE1EH33XefKJfLRV9fX4tf//jHP0RRNHUqf/jhhy1ek5ycLD7yyCOiKIriBx98IAYHB4tVVVXm53/55RdRJpOJ+fn5oiiKYkxMjPjss8+2OgYA4nPPPWf+uqqqSgQgrl+/3m6fk4ichzk3RORyEyZMwHvvvWfxWEhIiPn3Y8eOtXhu7NixOHz4MADg1KlTGDZsGHx9fc3Pjxs3DkajEadPn4YgCMjNzcXEiRPbHMPQoUPNv/f19UVAQAAKCws7+pGIyIUY3BCRy/n6+jbbJrIXb29vq65TKpUWXwuCAKPR6IghEZGDMeeGiLq83bt3N/t6wIABAIABAwbgyJEj0Gg05ud///13yGQy9O/fH/7+/khISEBaWppTx0xErsOVGyJyOa1Wi/z8fIvHFAoFwsLCAABr167FqFGjcPXVV2PVqlXYu3cvPvroIwDAnDlzsGTJEtx333148cUXUVRUhMceewz33nsvIiMjAQAvvvgiHn74YURERGD69OmorKzE77//jscee8y5H5SInILBDRG53IYNGxAdHW3xWP/+/ZGeng7AdJJp9erVePTRRxEdHY0vv/wSAwcOBAD4+Phg48aNWLRoEUaPHg0fHx/cdtttWL58ufle9913H2pra/HGG2/gT3/6E8LCwnD77bc77wMSkVMJoiiKrh4EEVFrBEHAd999h5tvvtnVQyGiboI5N0RERORWGNwQERGRW2HODRF1adw5JyJbceWGiIiI3AqDGyIiInIrDG6IiIjIrTC4ISIiIrfC4IaIiIjcCoMbIiIicisMboiIiMitMLghIiIit8LghoiIiNzK/wPpL6nNa7Y+UAAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAioAAAHHCAYAAACRAnNyAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAATrlJREFUeJzt3Xl4U2X6PvD7ZGnatE33FQotOy1QFChWcEF2EEHFUWBGRh0RKSqi/kbHUcuoA+oMoqIoLuB8FVBQcEUoyCLIvsku0FIKbSndkq5pmry/P9oGaktpQnJOWu7PdfUqOTk5efK0ys173vMeSQghQEREROSBVEoXQERERHQ5DCpERETksRhUiIiIyGMxqBAREZHHYlAhIiIij8WgQkRERB6LQYWIiIg8FoMKEREReSwGFSIiIvJYDCpE5HaSJCE1NdXh150+fRqSJGHx4sUur4mIWgYGFaJrxOLFiyFJEiRJwpYtWxo8L4RATEwMJEnC7bffrkCFztu4cSMkScKKFSuULoWIXIxBhega4+3tjSVLljTYvmnTJpw9exY6nU6BqoiIGsegQnSNGTVqFJYvX47q6up625csWYI+ffogMjJSocqIiBpiUCG6xkyYMAEFBQVIS0uzb6uqqsKKFSswceLERl9TVlaGp556CjExMdDpdOjatSv+85//4I83XzebzXjyyScRFhYGf39/3HHHHTh79myjxzx37hwefPBBREREQKfTISEhAZ988onrPmgj0tPTcc899yA4OBh6vR433HADfvjhhwb7vfPOO0hISIBer0dQUBD69u1bbxSqpKQEM2bMQGxsLHQ6HcLDwzF06FDs3bvXrfUTXYsYVIiuMbGxsUhOTsbSpUvt21avXg2j0Yj77ruvwf5CCNxxxx148803MWLECMydOxddu3bFM888g5kzZ9bb929/+xvmzZuHYcOGYc6cOdBqtRg9enSDY54/fx433HAD1q1bh+nTp+Ott95Cp06d8NBDD2HevHku/8x173njjTdizZo1mDZtGl599VVUVlbijjvuwMqVK+37ffjhh3j88ccRHx+PefPmYdasWejduzd27Nhh32fq1KlYsGAB7r77brz33nt4+umn4ePjg6NHj7qldqJrmiCia8KiRYsEALFr1y4xf/584e/vL8rLy4UQQtxzzz1i0KBBQggh2rdvL0aPHm1/3apV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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#UPLOAD TEST IMAGES\n",
        "\n",
        "uploaded = files.upload()\n",
        "test_filename = list(uploaded.keys())[0]\n",
        "\n",
        "# Load and process Test image\n",
        "img = image.load_img(test_filename, target_size=(64, 64))\n",
        "img_array = image.img_to_array(img)\n",
        "img_array = img_array.flatten() / 255.0\n",
        "img_array = np.expand_dims(img_array, axis=0)\n",
        "\n",
        "# Make a Waste Classification Prediction\n",
        "prediction = model.predict(img_array)\n",
        "predicted_class = np.argmax(prediction)  # Highest Probability\n",
        "\n",
        "# Waste Classes\n",
        "classes = ['Compost', 'General', 'Recycle']\n",
        "\n",
        "print(\"Classification of Waste:\", classes[predicted_class])"
      ],
      "metadata": {
        "id": "WNnPvqwnbqv6",
        "collapsed": true,
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 39
        },
        "outputId": "73322256-4381-497d-9c91-4bd0eff0e270"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "\n",
              "     <input type=\"file\" id=\"files-08f8aaa9-9961-4fd9-b8b1-300c0d0cc2cc\" name=\"files[]\" multiple disabled\n",
              "        style=\"border:none\" />\n",
              "     <output id=\"result-08f8aaa9-9961-4fd9-b8b1-300c0d0cc2cc\">\n",
              "      Upload widget is only available when the cell has been executed in the\n",
              "      current browser session. Please rerun this cell to enable.\n",
              "      </output>\n",
              "      <script>// Copyright 2017 Google LLC\n",
              "//\n",
              "// Licensed under the Apache License, Version 2.0 (the \"License\");\n",
              "// you may not use this file except in compliance with the License.\n",
              "// You may obtain a copy of the License at\n",
              "//\n",
              "//      http://www.apache.org/licenses/LICENSE-2.0\n",
              "//\n",
              "// Unless required by applicable law or agreed to in writing, software\n",
              "// distributed under the License is distributed on an \"AS IS\" BASIS,\n",
              "// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n",
              "// See the License for the specific language governing permissions and\n",
              "// limitations under the License.\n",
              "\n",
              "/**\n",
              " * @fileoverview Helpers for google.colab Python module.\n",
              " */\n",
              "(function(scope) {\n",
              "function span(text, styleAttributes = {}) {\n",
              "  const element = document.createElement('span');\n",
              "  element.textContent = text;\n",
              "  for (const key of Object.keys(styleAttributes)) {\n",
              "    element.style[key] = styleAttributes[key];\n",
              "  }\n",
              "  return element;\n",
              "}\n",
              "\n",
              "// Max number of bytes which will be uploaded at a time.\n",
              "const MAX_PAYLOAD_SIZE = 100 * 1024;\n",
              "\n",
              "function _uploadFiles(inputId, outputId) {\n",
              "  const steps = uploadFilesStep(inputId, outputId);\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  // Cache steps on the outputElement to make it available for the next call\n",
              "  // to uploadFilesContinue from Python.\n",
              "  outputElement.steps = steps;\n",
              "\n",
              "  return _uploadFilesContinue(outputId);\n",
              "}\n",
              "\n",
              "// This is roughly an async generator (not supported in the browser yet),\n",
              "// where there are multiple asynchronous steps and the Python side is going\n",
              "// to poll for completion of each step.\n",
              "// This uses a Promise to block the python side on completion of each step,\n",
              "// then passes the result of the previous step as the input to the next step.\n",
              "function _uploadFilesContinue(outputId) {\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  const steps = outputElement.steps;\n",
              "\n",
              "  const next = steps.next(outputElement.lastPromiseValue);\n",
              "  return Promise.resolve(next.value.promise).then((value) => {\n",
              "    // Cache the last promise value to make it available to the next\n",
              "    // step of the generator.\n",
              "    outputElement.lastPromiseValue = value;\n",
              "    return next.value.response;\n",
              "  });\n",
              "}\n",
              "\n",
              "/**\n",
              " * Generator function which is called between each async step of the upload\n",
              " * process.\n",
              " * @param {string} inputId Element ID of the input file picker element.\n",
              " * @param {string} outputId Element ID of the output display.\n",
              " * @return {!Iterable<!Object>} Iterable of next steps.\n",
              " */\n",
              "function* uploadFilesStep(inputId, outputId) {\n",
              "  const inputElement = document.getElementById(inputId);\n",
              "  inputElement.disabled = false;\n",
              "\n",
              "  const outputElement = document.getElementById(outputId);\n",
              "  outputElement.innerHTML = '';\n",
              "\n",
              "  const pickedPromise = new Promise((resolve) => {\n",
              "    inputElement.addEventListener('change', (e) => {\n",
              "      resolve(e.target.files);\n",
              "    });\n",
              "  });\n",
              "\n",
              "  const cancel = document.createElement('button');\n",
              "  inputElement.parentElement.appendChild(cancel);\n",
              "  cancel.textContent = 'Cancel upload';\n",
              "  const cancelPromise = new Promise((resolve) => {\n",
              "    cancel.onclick = () => {\n",
              "      resolve(null);\n",
              "    };\n",
              "  });\n",
              "\n",
              "  // Wait for the user to pick the files.\n",
              "  const files = yield {\n",
              "    promise: Promise.race([pickedPromise, cancelPromise]),\n",
              "    response: {\n",
              "      action: 'starting',\n",
              "    }\n",
              "  };\n",
              "\n",
              "  cancel.remove();\n",
              "\n",
              "  // Disable the input element since further picks are not allowed.\n",
              "  inputElement.disabled = true;\n",
              "\n",
              "  if (!files) {\n",
              "    return {\n",
              "      response: {\n",
              "        action: 'complete',\n",
              "      }\n",
              "    };\n",
              "  }\n",
              "\n",
              "  for (const file of files) {\n",
              "    const li = document.createElement('li');\n",
              "    li.append(span(file.name, {fontWeight: 'bold'}));\n",
              "    li.append(span(\n",
              "        `(${file.type || 'n/a'}) - ${file.size} bytes, ` +\n",
              "        `last modified: ${\n",
              "            file.lastModifiedDate ? file.lastModifiedDate.toLocaleDateString() :\n",
              "                                    'n/a'} - `));\n",
              "    const percent = span('0% done');\n",
              "    li.appendChild(percent);\n",
              "\n",
              "    outputElement.appendChild(li);\n",
              "\n",
              "    const fileDataPromise = new Promise((resolve) => {\n",
              "      const reader = new FileReader();\n",
              "      reader.onload = (e) => {\n",
              "        resolve(e.target.result);\n",
              "      };\n",
              "      reader.readAsArrayBuffer(file);\n",
              "    });\n",
              "    // Wait for the data to be ready.\n",
              "    let fileData = yield {\n",
              "      promise: fileDataPromise,\n",
              "      response: {\n",
              "        action: 'continue',\n",
              "      }\n",
              "    };\n",
              "\n",
              "    // Use a chunked sending to avoid message size limits. See b/62115660.\n",
              "    let position = 0;\n",
              "    do {\n",
              "      const length = Math.min(fileData.byteLength - position, MAX_PAYLOAD_SIZE);\n",
              "      const chunk = new Uint8Array(fileData, position, length);\n",
              "      position += length;\n",
              "\n",
              "      const base64 = btoa(String.fromCharCode.apply(null, chunk));\n",
              "      yield {\n",
              "        response: {\n",
              "          action: 'append',\n",
              "          file: file.name,\n",
              "          data: base64,\n",
              "        },\n",
              "      };\n",
              "\n",
              "      let percentDone = fileData.byteLength === 0 ?\n",
              "          100 :\n",
              "          Math.round((position / fileData.byteLength) * 100);\n",
              "      percent.textContent = `${percentDone}% done`;\n",
              "\n",
              "    } while (position < fileData.byteLength);\n",
              "  }\n",
              "\n",
              "  // All done.\n",
              "  yield {\n",
              "    response: {\n",
              "      action: 'complete',\n",
              "    }\n",
              "  };\n",
              "}\n",
              "\n",
              "scope.google = scope.google || {};\n",
              "scope.google.colab = scope.google.colab || {};\n",
              "scope.google.colab._files = {\n",
              "  _uploadFiles,\n",
              "  _uploadFilesContinue,\n",
              "};\n",
              "})(self);\n",
              "</script> "
            ]
          },
          "metadata": {}
        }
      ]
    }
  ]
}