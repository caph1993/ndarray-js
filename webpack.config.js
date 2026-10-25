//@ts-check
const path = require('path');
const licensePlugin = require('webpack-license-plugin');
const LicensePlugin = licensePlugin.default || licensePlugin; // the export differs between versions
const UglifyJsPlugin = require('uglifyjs-webpack-plugin');

// const fs = require('fs');
// const licenseText = fs.readFileSync(path.resolve(__dirname, './LICENSE'), 'utf8');
// const licenseText = 'numpy-js - copyright 2023 Carlos Pinzón <caph1993@gmail.com>';

module.exports = {
  module: {
    rules: [
      {
        test: /\.ts?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
  output: {
    publicPath: '',
    filename: 'index.js',
    path: path.resolve(__dirname, 'dist'),
    globalObject: 'this', // works in the browser and in node
    libraryTarget: 'umd', // Fix: "Uncaught ReferenceError: exports is not defined".
  },
  plugins: [
    new LicensePlugin(),
    // new webpack.BannerPlugin({ banner: licenseText, raw: true }),
  ],
  devtool: 'source-map',
  optimization: {
    minimizer: [new UglifyJsPlugin({ sourceMap: true })],
  },
  mode: "production",
};