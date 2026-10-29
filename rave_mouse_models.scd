/*
RAVE + Ndef: the mouse (left-right) plays the role of "proximity" (0..1).
Works with every 48 kHz .ts model found in the models folder, switchable while running.

Requirements: SuperCollider 3.13+, nn.ar installed, models in the folder set below.
Not tested by me, I could not run SuperCollider. If something fails, send the post window text.

Volume:   Ndef(\rave).set(\amp, 6);     // output level (Limiter protects at 0.8)
          Ndef(\rave).set(\inGain, 3);  // input level into the encoder
Models:   ~use.(0);  ~use.(5);  ~next.value;   // by index from the list printed in step 1
*/

// 1. Setup (run once). Prints the list of models with their index.
(
s.options.sampleRate = 48000;   // models with r44100 in the name are skipped
s.options.numInputBusChannels = 2;
s.options.numOutputBusChannels = 2;
s.waitForBoot {
	~modelDir = "/Users/marcellolussana/Documents/Software/RAVE_SC/models/";

	~files = PathName(~modelDir).files
		.select { |f| f.extension == "ts" }
		.reject { |f| f.fileName.contains("r44100") }
		.sort { |a, b| a.fileName < b.fileName };

	// latent size from the file name: "..._z16.ts" or "..._18d.ts"
	~latentSize = { |name|
		var m = name.findRegexp("[_-]z(\\d+)\\.ts");
		if(m.isEmpty) { m = name.findRegexp("_(\\d+)d\\.ts") };
		if(m.isEmpty) { nil } { m[1][1].asInteger };
	};

	~loaded = Set.new;
	~idx = 0;

	"Models found:".postln;
	~files.do { |f, i|
		"  %: %  (latent size %)".format(i, f.fileName, ~latentSize.(f.fileName)).postln;
	};
};
)

// 2. Mouse as control signal (stands in for proximity, 0..1)
(
Ndef(\prox, { MouseX.kr(0, 1).lag(0.2) });
)

// 3. Settings and the function that loads and plays a model
(
~range = 4;        // how far the latent vector is pushed (try 2 to 6)
~dim   = nil;      // nil = push ALL latent dimensions in a fixed random direction (more variation)
                   // a number (e.g. 0) = push only that single dimension
~seed  = 1;        // changes the random direction when ~dim is nil
~input = \noise;   // \noise, \mic or \file

// ~buf = Buffer.read(s, "/path/to/some/sound.wav");   // needed only for ~input = \file

~inputs = (
	noise: { HPF.ar(PinkNoise.ar, 150) * LFNoise2.kr(0.4).range(0.1, 1) },
	mic:   { SoundIn.ar(0) },   // use headphones, otherwise feedback
	file:  { PlayBuf.ar(1, ~buf, BufRateScale.kr(~buf), loop: 1) }
);

~use = { |index|
	fork {
		var file = ~files[index];
		var name = ("m" ++ index).asSymbol;
		var z = ~latentSize.(file.fileName);
		var dir;

		if(z.isNil) {
			"Could not read the latent size from %, skipping".format(file.fileName).warn;
		} {
			if(~loaded.includes(name).not) {
				NN.load(name, file.fullPath);
				s.sync;
				~loaded.add(name);
			};

			// direction in latent space that the mouse moves along
			if(~dim.isNil) {
				thisThread.randSeed = ~seed;
				dir = Array.fill(z) { 1.0.rand2 };
				dir = dir / dir.squared.mean.sqrt;   // average size 1 per dimension
			} {
				dir = Array.fill(z) { |i| if(i == ~dim) { 1 } { 0 } };
			};

			~idx = index;
			Ndef(\rave).fadeTime = 1;
			Ndef(\rave, {
				var prox = \prox.kr(0);
				var amp = \amp.kr(3);
				var inGain = \inGain.kr(1);
				var in = ~inputs[~input].value * inGain;
				var lat = NN(name, \encode).ar(in);
				var shift = dir * prox.linlin(0, 1, ~range.neg, ~range);
				var out = NN(name, \decode).ar(lat + shift);
				Limiter.ar(Pan2.ar(out, 0, amp), 0.8);
			});
			Ndef(\rave).map(\prox, Ndef(\prox));
			Ndef(\rave).play;
			"Playing %: % (latent size %)".format(index, file.fileName, z).postln;
		};
	};
};

~next = { ~use.((~idx + 1) % ~files.size) };
)

// 4. Play and explore
~use.(8);          // first model in the list
~next.value;       // go to the next model
// ~use.(3);

// louder / quieter
// Ndef(\rave).set(\amp, 6);
// Ndef(\rave).set(\inGain, 3);

// after changing ~range, ~dim, ~seed or ~input, run ~use.(~idx) again

// Stop
// Ndef(\rave).free; Ndef(\prox).free;
