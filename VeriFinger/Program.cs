using System;
using Neurotec.Biometrics;
using Neurotec.Biometrics.Client;
using Neurotec.Licensing;
using Neurotec.Images;

namespace Neurotec.Tutorials
{
	class Program
	{
		private static int Usage()
		{
			Console.WriteLine("usage:");
			Console.WriteLine("\t{0} 1_1.tif 1_2.tif", TutorialUtils.GetAssemblyName());
			Console.WriteLine();

			return 2;
		}

		static int Main(string[] args)
		{
			TutorialUtils.PrintTutorialHeader(args);

			if (args.Length < 2)
			{
				return Usage();
			}

			//=========================================================================
			// CHOOSE LICENCES !!!
			//=========================================================================
			// ONE of the below listed "licenses" lines is required for unlocking this sample's functionality. Choose licenses that you currently have on your device.
			// If you are using a TRIAL version - choose any of them.

			//const string licenses = "FingerMatcher,FingerExtractor";
			const string licenses = "FingerMatcher,FingerClient";
			//const string licenses = "FingerFastMatcher,FingerFastExtractor";

			//=========================================================================

			//=========================================================================
			// TRIAL MODE
			//=========================================================================
			// Below code line determines whether TRIAL is enabled or not. To use purchased licenses, don't use below code line.
			// GetTrialModeFlag() method takes value from "Bin/Licenses/TrialFlag.txt" file. So to easily change mode for all our examples, modify that file.
			// Also you can just set TRUE to "TrialMode" property in code.

			//NLicenseManager.TrialMode = TutorialUtils.GetTrialModeFlag();

			//Console.WriteLine("Trial mode: " + NLicenseManager.TrialMode);

			//=========================================================================

			try
			{
				// Obtain licenses
				if (!NLicense.Obtain("/local", 5000, licenses))
				{
					throw new ApplicationException(string.Format("Could not obtain licenses: {0}", licenses));
				}

				Console.Write("Command line arugments: {0} {1}", args[0], args[1]);
				using (var biometricClient = new NBiometricClient())
				// Create subjects with finger object
				//using (NSubject referenceSubject = CreateSubject(args[0], args[0]))
				//using (NSubject candidateSubject = CreateSubject(args[1], args[1]))
				//using (NSubject referenceSubject = CreateSubject(args[0], args[0]))
				//using (NSubject candidateSubject = CreateSubject(args[1], args[1]))

				{
					NImage img0 = NImage.FromFile(args[0]);
					NImage img1 = NImage.FromFile(args[1]);

					img1.HorzResolution = 500;
					img1.VertResolution = 500;
					img1.ResolutionIsAspectRatio = false;

					if (img1.HorzResolution < 250 || img1.VertResolution < 250)
					{
						img1.ResolutionIsAspectRatio = false;
						if (img1.HorzResolution < 250)
						{
							img1.HorzResolution = 500;
						}
						if (img1.VertResolution < 250)
						{
							img1.VertResolution = 500;
						}
					}
					NSubject candidateSubject = new NSubject();
					var finger1 = new NFinger { Image = img1 };
					candidateSubject.Fingers.Add(finger1);

					img0.HorzResolution = 500;
					img0.VertResolution = 500;
					img0.ResolutionIsAspectRatio = false;

					if (img0.HorzResolution < 250 || img0.VertResolution < 250)
					{
						img0.ResolutionIsAspectRatio = false;
						if (img0.HorzResolution < 250)
						{
							img0.HorzResolution = 500;
						}
						if (img0.VertResolution < 250)
						{
							img0.VertResolution = 500;
						}
					}

					NSubject referenceSubject = new NSubject();
					var finger0 = new NFinger { Image = img0 };
					referenceSubject.Fingers.Add(finger0);

					// Set matching threshold
					biometricClient.MatchingThreshold = 48;

					// Set matching speed
					biometricClient.FingersMatchingSpeed = NMatchingSpeed.Low;

					// Verify subjects
					NBiometricStatus status = biometricClient.Verify(referenceSubject, candidateSubject);
					if (status == NBiometricStatus.Ok || status == NBiometricStatus.MatchNotFound)
					{
						int score = referenceSubject.MatchingResults[0].Score;
						Console.Write("Image scored {0}, verification ", score);
						Console.WriteLine(status == NBiometricStatus.Ok ? "succeeded" : "failed");
					}
					else
					{
						Console.Write("Verification failed. Status: {0}", status);
						return -1;
					}
				}

				return 0;
			}
			catch (Exception ex)
			{
				return TutorialUtils.PrintException(ex);
			}
		}

		//private static NSubject CreateSubject(string fileName, string subjectId)
		//{
		//	var subject = new NSubject {Id = subjectId};
		//	var finger = new NFinger { FileName = fileName };
		//	subject.Fingers.Add(finger);
		//	return subject;
		//}
		//private static NSubject CreateSubject(string fileName)
		//{
		//	var subject = new NSubject();
		//	subject.SetTemplateBuffer(new NBuffer(File.ReadAllBytes(fileName)));
		//	subject.Id = fileName;

		//	return subject;
		//}
	}
}
