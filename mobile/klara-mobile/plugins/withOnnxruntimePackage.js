// Android settings this app needs on every prebuild:
//  * onnxruntime-react-native is a legacy native module that autolinking misses, so
//    NativeModules.Onnxruntime was null at startup ("Cannot read property 'install' of null").
//    Its package is registered by hand.
//  * arm64 only, and two llama.rn CPU variants instead of six: the full set filled the disk while
//    compiling llama.cpp from source (llama.rn's rc forces a source build).
const { withMainApplication, withGradleProperties, withAppBuildGradle } = require('@expo/config-plugins');

const PROPS = {
  reactNativeArchitectures: 'arm64-v8a',
  rnllamaVariants: 'rnllama;rnllama_v8_2_dotprod',
};

module.exports = function withKlaraAndroid(config) {
  config = withMainApplication(config, (cfg) => {
    let src = cfg.modResults.contents;
    if (!src.includes('OnnxruntimePackage()')) {
      src = src.replace(
        /PackageList\(this\)\.packages\.apply \{/,
        'PackageList(this).packages.apply {\n              add(ai.onnxruntime.reactnative.OnnxruntimePackage())',
      );
    }
    cfg.modResults.contents = src;
    return cfg;
  });
  // KLARA_DEBUGGABLE=1 at prebuild: a release build adb can `run-as`, to copy models in for testing.
  if (process.env.KLARA_DEBUGGABLE === '1') {
    config = withAppBuildGradle(config, (cfg) => {
      if (!cfg.modResults.contents.includes('debuggable true // klara')) {
        cfg.modResults.contents = cfg.modResults.contents.replace(
          /buildTypes \{([\s\S]*?)release \{/, (m) => m + '\n            debuggable true // klara');
      }
      return cfg;
    });
  }
  return withGradleProperties(config, (cfg) => {
    for (const [key, value] of Object.entries(PROPS)) {
      cfg.modResults = cfg.modResults.filter((p) => !(p.type === 'property' && p.key === key));
      cfg.modResults.push({ type: 'property', key, value });
    }
    return cfg;
  });
};
