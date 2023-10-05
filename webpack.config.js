//@ts-check
const path = require('path');
// const CopyWebpackPlugin = require('copy-webpack-plugin');const fs = require('fs');
const webpack = require('webpack');
const LicensePlugin = require('webpack-license-plugin').default;


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
    // new webpack.BannerPlugin({ banner: licenseText, raw: true }),
    new LicensePlugin(),
  ],
  mode: "production",
  // devServer: {
  //   static: [
  //     path.resolve(__dirname, 'dist'),
  //     path.resolve(__dirname),
  //   ],
  // },
  // plugins: [
  //   new CopyWebpackPlugin({
  //     patterns: [
  //       { from: 'src/index.html', to: 'index.html' },
  //       // { from: 'put.js', to: 'put.js' },
  //       // { from: 'putTools.js', to: 'putTools.js' },
  //       // { from: 'cpTools.js', to: 'cpTools.js' },
  //     ],
  //   }),
  // ],
};