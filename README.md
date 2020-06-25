# Trellis: Deep Hedging and Deep Pricing

Trellis is a deep hedging and deep pricing framework with the primary purpose of furthering research into the use of neural networks as a replacement for classical analytical methods for pricing and hedging financial instruments.

The project is built in Python on top of TensorFlow and Keras.

Trellis was originally developed by engineers at [Beacon Platform](https://beacon.io) to conduct research into the deep hedging technique and foster collaboration within the finance industry and with academia.

If you are using this in your own project or research, we would be interested to hear from you.

## Installation

Trellis is available on PyPi, simply install with `pip`.

    pip install beacon-trellis

Note that only TensorFlow version 2.1.0 is currently supported.

To use, simply

    import trellis

See `dh_european_option.py` and `dh_variable_annuity.py` for examples of how to use the models and visualisations provided.

## Coming Soon

- Example Jupyter Notebooks
- TensorFlow 2.2.0 support
- ReadTheDocs documentation

## Contribution guidelines

If you want to contribute to the project, be sure to review the [contribution guidelines](CONTRIBUTING.md).

## Contact Information

- For technical questions or bugs please [log an issue](https://github.com/Beacon-Platform/trellis/issues).
- For business enquiries, please contact [Beacon Platform](https://www.beacon.io/contact) directly.
- [Beacon Platform Twitter](https://twitter.com/PlatformBeacon).
- [Beacon Platform LinkedIn](https://www.linkedin.com/company/beacon-platform-inc/).
- Maintained by [Benjamin Pryke](https://github.com/benpryke).

## License

[MIT](LICENSE)
