import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  private var embedService: EmbedService?
  private var ragService: RAGService?
  
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    let result = super.application(application, didFinishLaunchingWithOptions: launchOptions)
    
    GeneratedPluginRegistrant.register(with: self)
    
    // Register embed service using FlutterEngine
    if let controller = window?.rootViewController as? FlutterViewController {
      embedService = EmbedService(binaryMessenger: controller.binaryMessenger)
      ragService = RAGService(binaryMessenger: controller.binaryMessenger)
    }
    
    return result
  }
}
