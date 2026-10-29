/*
Minimal RAVE + Ndef example: a birds model, one latent dimension controlled by the mouse.
The mouse plays the role of the "proximity" value (0..1).

Requirements:
- SuperCollider 3.13 or newer
- nn.ar extension (https://github.com/elgiano/nn.ar), installed in the Extensions folder with its nn_tilde backend
- A RAVE model exported as .ts (TorchScript). Set the path below.

Note: written from my knowledge of the nn.ar API, not tested here. If a method name
fails, check the nn.ar help files and run NN(\rave).describe.
*/
Platform.userExtensionDir.postln;
// 1. Boot and load the model
(
s.options.sampleRate = 48000;   // the bird models are 48 kHz
s.options.numInputBusChannels = 2;
s.options.numOutputBusChannels = 2;
s.waitForBoot {
	~modelPath = "/Users/marcellolussana/Documents/Software/RAVE_SC/models/birds_dawnchorus_b2048_r48000_z8.ts";
	// other options: birds_pluma_b2048_r48000_z12.ts (z12), birds_motherbird_b2048_r48000_z16.ts (z16)
	NN.load(\rave, ~modelPath);
	s.sync;
	NN(\rave).describe;   // posts the methods and the latent size
};
)

// 2. Mouse as a control signal (stands in for proximity, 0..1)
(
Ndef(\prox, { MouseX.kr(0, 1).lag(0.2) });   // lag smooths jumps
)

// 3. RAVE in an Ndef: encode -> shift one latent dimension -> decode
(
~nDims = 8;    // latent size: dawnchorus 8, pluma 12, motherbird 16 (check with describe)
~dim   = 0;    // which latent dimension the mouse moves
~range = 3;    // how far it moves (try 1 to 4)

Ndef(\rave, {
	var prox = \prox.kr(0);
	var in = SoundIn.ar(0);   // mic input. Alternative: PlayBuf.ar(1, ~buf, loop: 1)
	var z = NN(\rave, \encode).ar(in);

	// offset only on one dimension, 0 -> -range, 1 -> +range
	var offset = Array.fill(~nDims, { |i|
		if(i == ~dim) { prox.linlin(0, 1, ~range.neg, ~range) } { 0 }
	});

	var out = NN(\rave, \decode).ar(z + offset);
	Pan2.ar(out );
});

// connect the mouse signal to the \prox control
Ndef(\rave).map(\prox, Ndef(\prox));
Ndef(\rave).play;
)

// 4. Experiment while it runs
// Change the dimension (re-evaluate block 3 after changing ~dim), or set values by hand:
// Ndef(\rave).unmap(\prox); Ndef(\rave).set(\prox, 0.5);
// Ndef(\rave).map(\prox, Ndef(\prox));   // back to the mouse

// Stop
// Ndef(\rave).free; Ndef(\prox).free;
