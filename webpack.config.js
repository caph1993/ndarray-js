//@ts-check
const path = require('path');
// const CopyWebpackPlugin = require('copy-webpack-plugin');const fs = require('fs');
const LicensePlugin = require('webpack-license-plugin').default;
const UglifyJsPlugin = require('uglifyjs-webpack-plugin');


const fs = require('fs');
// const licenseText = fs.readFileSync(path.resolve(__dirname, './LICENSE'), 'utf8');

const licenseText = 'numpy-js - copyright 2023 Carlos Pinzón <caph1993@gmail.com>';

module.exports = {
  entry: './src/index.js',
  output: {
    publicPath: '',
    filename: 'caph1993-numpy-js.js',
    path: path.resolve(__dirname, 'dist'),
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