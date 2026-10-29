/*
Overlapping grains controlled by ONE value (0..1), optionally coloured by a RAVE model.

Two behaviours (menu "behaviour"):
- pad:        a calm ambient texture. Value up = more present (louder), brighter (the low-pass opens and the grains
              come from brighter parts of the recording) and denser (more overlapping long grains).
- percussive: short hits that get faster and denser when the value goes up (from about 0.6 to 14 grains per second,
              grains from 0.5 s down to 0.1 s). Above value 0.5 some hits are doubled.

Grains are cut from a real recording. The recording is analysed once with make_corpus.py (0.5 s regions, the
table is saved next to the recording). Pad grains are taken from calm regions, percussive grains from regions with
bursts and sound events.

RAVE sources (menu "source"): (0) the grain cloud goes through the model, (1) the recording itself, looped, goes through the
model (plain resynthesis; the grains are not used), (2) model only: no input, the decoder is driven by latents that wander
slowly around the mean of the recording (statistics from rave_latent_stats.py). In 1 and 2 the value acts through the latent
shift, the low-pass and the level; in 2 with "percussive" the output is gated into hits that get faster with the value.

RAVE: the grain cloud can be sent through a RAVE model (encode, optional shift of the latent vector by the value,
decode). "RAVE mix" 0 = grains only, 1 = only the model output. The model hears the grains at a constant level;
the loudness that belongs to the value is applied after the model. Needs the nn.ar extension and the models folder.

I could not run SuperCollider, so this is untested. If something fails, send me the post window text.
*/

(
s.waitForBoot {
	// clean up an earlier run, so that this file can be evaluated again without restarting the server
	if(~grStop.notNil) { ~grStop.value };
	if(~grSweepR.notNil) { ~grSweepR.stop };
	if(~grPoll.notNil) { ~grPoll.stop };
	if(~grWin.notNil and: { ~grWin.isClosed.not }) { ~grWin.close };   // its onClose also frees the output stage
	if(~grBuf.notNil) { ~grBuf.free };
	if(~grGrp.notNil) { ~grGrp.free };
	~grDir      = "/Users/marcellolussana/Documents/Software/RAVE_SC/";
	~grAudioDir = ~grDir ++ "audio/";
	~grModelDir = ~grDir ++ "models/";
	~grPy       = ~grDir ++ ".venv/bin/python";
	~grScript   = ~grDir ++ "make_corpus.py";
	~grRegion   = "0.5";    // length (s) of the regions used as grain sources; the table is <name>_seg0.5.csv
	~grQ = { |p| "\"" ++ p ++ "\"" };

	// state
	~grMode = 0;            // 0 = pad, 1 = percussive
	~grV = 0.3;
	~grDens = 1.0;          // multiplier of the grain rate
	~grLen = 1.0;           // multiplier of the grain length
	~grVol = 0.8;
	~grShift = 1.0;         // latent shift at value 0 and 1 (towards -/+ along the direction; 1 = the strength set in the direction file)
	~grSoft = 0.5;          // softness of the percussive attack: 0.004 s + softness * 0.03 s
	~grCeil = 9000;         // highest cutoff of the low-pass (Hz)
	~grTame = 4;            // dB taken off above 3.5 kHz
	~grSmooth = 0.05;       // smoothing of the latent vector (s)
	~grAvoid = 1;           // 1 = do not use the harshest source regions
	~grDirMode = 0;         // 0 = computed direction (louder and brighter), 1 = random
	~grComp = 1;            // 1 = the model output is brought to the level of the grains
	~grSrc = 0;             // RAVE source: 0 = the grain cloud, 1 = the recording itself, 2 = model only (no input)
	~grWalk = 0.5;          // model only: size of the latent wander (in spreads of the recording)
	~grWalkRate = 0.3;      // model only: speed of the wander (Hz)
	~grMakeup = [14, 10];   // dB added after the model (pad, percussive): the grains themselves stay at a level the model likes
	~grRows = [];
	~grBuf = nil; ~grFile = nil; ~grSR = 44100; ~grCh = 2; ~grFrames = 1;
	~grPlaying = false;
	~grR = nil; ~grSweepR = nil;
	~grRecent = [];
	~grTried = nil;
	~grZ = 8;
	~grLoaded = ~grLoaded ?? { Set.new };   // models stay loaded on the server between runs
	~grModelIdx = nil;
	~grBus = ~grBus ?? { Bus.audio(s, 2) };
	~grLatBus = ~grLatBus ?? { Bus.control(s, 64) };   // 0..31 latents from the encoder, 32..63 after the shift
	~grLat = nil; ~grHist = []; ~grPts = nil; ~grUv = nil;
	if(~grPoll.notNil) { ~grPoll.stop; ~grPoll = nil };
	~grGrp = Group.head(s.defaultGroup);
	~grValSl = nil; ~grInfo = nil; ~grStatus = nil; ~grPlayBtn = nil;   // widgets of an earlier run may be closed

	// ---- grain players: pad = smooth (sine) window, percussive = fast attack, exponential-like decay
	[[\grPad, \sine], [\grPerc, [2, -6]]].do { |pair|
		SynthDef((pair[0] ++ "2").asSymbol, { |buf, start = 0, glen = 1, atk = 0.5, amp = 0.2, pan = 0, out = 0|
			var env = EnvGen.kr(Env([0, 1, 0], [atk, (glen - atk).max(0.005)], pair[1]), doneAction: 2);
			var sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf), startPos: start);
			Out.ar(out, Balance2.ar(sig[0], sig[1], pan) * env * amp);
		}).add;
		SynthDef((pair[0] ++ "1").asSymbol, { |buf, start = 0, glen = 1, atk = 0.5, amp = 0.2, pan = 0, out = 0|
			var env = EnvGen.kr(Env([0, 1, 0], [atk, (glen - atk).max(0.005)], pair[1]), doneAction: 2);
			var sig = PlayBuf.ar(1, buf, BufRateScale.kr(buf), startPos: start);
			Out.ar(out, Pan2.ar(sig, pan) * env * amp);
		}).add;
	};
	s.sync;

	// ---- recordings
	~grAudioFiles = {
		var isAudio = { |p| var q = p.toLower; [".wav", ".flac", ".aif", ".aiff"].any { |e| q.endsWith(e) } };
		var raw = (~grAudioDir ++ "*").pathMatch;
		var list = raw.select(isAudio);
		if(list.isEmpty) {
			list = ("ls -1 " ++ ~grQ.(~grAudioDir)).unixCmdGetStdOut.split($\n)
				.reject { |l| l.isEmpty }.collect { |l| ~grAudioDir ++ l }.select(isAudio);
		};
		"recordings found in %: % (the folder search matched % entries)".format(~grAudioDir, list.size, raw.size).postln;
		list.sort
	};
	~grFileList = ~grAudioFiles.value;
	~grTable = { ~grFile.dirname +/+ (~grFile.basename.splitext[0] ++ "_seg" ++ ~grRegion ++ ".csv") };

	// ---- models
	~grFiles = PathName(~grModelDir).files
		.select { |f| f.extension == "ts" }
		.reject { |f| f.fileName.contains("r44100") }
		.sort { |a, b| a.fileName < b.fileName };
	~grLatentSize = { |name|
		var m = name.findRegexp("[_-]z(\\d+)\\.ts");
		if(m.isEmpty) { m = name.findRegexp("_(\\d+)d\\.ts") };
		if(m.isEmpty) { nil } { m[1][1].asInteger };
	};

	// ---- ranks (0..1, equal values share the mean rank)
	~grRanks = { |vals|
		var n = vals.size, ord = vals.order, rk = Array.fill(n, 0.0), i = 0, j;
		while { i < n } {
			j = i;
			while { (j + 1 < n) and: { vals[ord[j + 1]] == vals[ord[i]] } } { j = j + 1 };
			(i..j).do { |k| rk[ord[k]] = ((i + j) / 2) / (n - 1).max(1) };
			i = j + 1;
		};
		rk
	};

	// ---- two pools of regions: calm ones for the pad, bursty ones for the percussive behaviour
	~grPrepare = {
		var n = ~grRows.size;
		var rAct = ~grRanks.(~grRows.collect { |r| r[6] });
		var rOn  = ~grRanks.(~grRows.collect { |r| r[9] });
		var rRms = ~grRanks.(~grRows.collect { |r| r[2] });
		var rCh  = ~grRanks.(~grRows.collect { |r| r[7] });   // brightness of the hiss: the highest ones are the harsh regions
		var all = (0..(n - 1));
		var pad, perc, score;
		pad = all.select { |i| (rAct[i] <= 0.6) and: { (rOn[i] <= 0.7) and: { (~grAvoid == 0) or: { rCh[i] <= 0.9 } } } };
		if(pad.size < 20) { pad = all.sort { |a, b| rAct[a] <= rAct[b] }.keep(max(20, (n div: 3))) };
		score = all.collect { |i| rAct[i] + rOn[i] + (0.5 * rRms[i]) - (if((~grAvoid == 1) and: { rCh[i] > 0.9 }) { 100 } { 0 }) };
		perc = all.sort { |a, b| score[a] >= score[b] }.keep(max(20, (n * 0.35).asInteger));
		~grPadIdx  = pad;
		~grPadKey  = ~grRanks.(pad.collect { |i| ~grRows[i][8] });    // brightness: energy above 2 kHz
		~grPercIdx = perc;
		~grPercKey = ~grRanks.(perc.collect { |i| ~grRows[i][2] });   // loudness
		"pools: % calm regions, % bursty regions (of %)".format(pad.size, perc.size, n).postln;
		~grMakePts.value;
	};
	// positions of all regions in the plot: x = rank of the brightness (energy above 2 kHz), y = rank of the burstiness
	// (activity). Ranks spread the dots over the whole panel (the raw values pile up near the bottom).
	~grMakePts = {
		var rx = ~grRanks.(~grRows.collect { |r| r[8] });
		var ry = ~grRanks.(~grRows.collect { |r| r[6] });
		~grPts = rx.collect { |x, i| [x, ry[i]] };
	};

	// next region for this value: the pool is ordered by brightness (pad) or loudness (percussive)
	~grPick = { |mode, v|
		var idx = if(mode == 0) { ~grPadIdx } { ~grPercIdx };
		var key = if(mode == 0) { ~grPadKey } { ~grPercKey };
		var width = if(mode == 0) { 0.15 } { 0.35 };
		var near = (0..(idx.size - 1)).select { |j| (key[j] - v).abs <= width };
		var cands, j;
		if(near.size < 6) {
			near = (0..(idx.size - 1)).sort { |a, b| (key[a] - v).abs <= (key[b] - v).abs }.keep(6);
		};
		cands = near.reject { |q|
			~grRecent.any { |r| (~grRows[idx[q]][0] - ~grRows[r][0]).abs < 0.3 }
		};
		if(cands.isEmpty) { cands = near };
		j = idx[cands.choose];
		~grRecent = (~grRecent ++ [j]).keep(-8);
		j
	};

	// how the value shapes the sound
	~grParams = { |mode, v|
		var rate, glen, cut, db, gain, ov;
		if(mode == 0) {
			rate = v.linexp(0, 1, 1.5, 9) * ~grDens;
			glen = 1.6 * ~grLen;
			cut  = v.linexp(0, 1, 500, ~grCeil);
			db   = v.linlin(0, 1, -26, -4);
		} {
			rate = v.linexp(0, 1, 0.6, 14) * ~grDens;
			glen = v.linexp(0, 1, 0.5, 0.1) * ~grLen;
			cut  = v.linexp(0, 1, min(3000, ~grCeil * 0.6), ~grCeil);
			db   = v.linlin(0, 1, -16, -3);
		};
		ov = rate * glen;
		gain = if(mode == 0) { 1 / max(1, ov * 0.5).sqrt } { 0.8 };   // pad: many overlapping grains do not add up to a louder sound
		(rate: rate, glen: glen, cut: cut, db: db, gain: gain, overlap: ov)
	};

	~grGrain = { |i, p, mode|
		var r = ~grRows[i];
		var glen = p[\glen];
		var off = (r[1] - glen).max(0).rand;
		var startF = ((r[0] + off) * ~grSR).min(~grFrames - (glen * ~grSR) - 2).max(0);
		var norm = ((if(mode == 0) { -24 } { -20 }) - r[2]).dbamp.clip(0.2, 6);   // grains of similar level
		var pan = if(mode == 0) { 0.6.rand2 } { 0.8.rand2 };
		var name = ((if(mode == 0) { "grPad" } { "grPerc" }) ++ ~grCh.min(2).asString).asSymbol;
		~grHist = (~grHist ++ [[i, Main.elapsedTime]]).keep(-30);
		Synth(name, [\buf, ~grBuf, \start, startF, \glen, glen,
			\atk, if(mode == 0) { glen * 0.5 } { 0.004 + (~grSoft * 0.03) },
			\amp, p[\gain] * norm * rrand(0.8, 1.0), \pan, pan, \out, ~grBus.index], ~grGrp);
	};

	~grStop = {
		~grPlaying = false;
		if(~grR.notNil) { ~grR.stop; ~grR = nil };
	};
	~grStart = {
		~grStop.value;
		if((~grRows.size > 10) and: { ~grBuf.notNil and: { ~grModelIdx.isNil or: { ~grSrc == 0 } } }) {
			~grPlaying = true;
			~grR = Routine({
				while { ~grPlaying } {
					var mode = ~grMode;
					var v = ~grV;
					var p = ~grParams.(mode, v);
					var wait;
					~grGrain.(~grPick.(mode, v), p, mode);
					if((mode == 1) and: { (v > 0.5) and: { 1.0.rand < ((v - 0.5) * 1.4) } }) {
						SystemClock.sched(rrand(0.01, 0.04), { ~grGrain.(~grPick.(mode, v), p, mode); nil });
					};
					wait = (1 / p[\rate]) * rrand(0.5, 1.5);
					wait.max(0.01).wait;
				};
			}).play(SystemClock);
		};
	};

	// ---- output stage: low-pass (opens with the value), optional RAVE, loudness (follows the value)
	~grBuild = {
		var name = if(~grModelIdx.notNil) { ("g" ++ ~grModelIdx).asSymbol };
		var z = ~grZ;
		var src = if(name.isNil) { 0 } { ~grSrc };
		var mode = ~grMode;
		if((src == 1) and: { ~grBuf.isNil }) { src = 0 };
		Ndef(\grOut).fadeTime = 0.4;
		Ndef(\grOut).unset(\dir);
		Ndef(\grOut).unset(\lmean);
		Ndef(\grOut).unset(\lstd);
		Ndef(\grOut, {
			var cut = \cut.kr(8000, 0.1);
			var amp = \amp.kr(0.3, 0.1);
			var v = \v.kr(0.5, 0.05);
			var wet = \wet.kr(1);
			var inGain = \inGain.kr(1);
			var shift = \shift.kr(0.5);
			var comp = \comp.kr(1);
			var tame = \tame.kr(4);
			var smooth = \smooth.kr(0.05);
			var walkSize = \walk.kr(0.5);
			var walkRate = \walkRate.kr(0.3);
			var hitRate = \hitRate.kr(2);
			var hitLen = \hitLen.kr(0.3);
			var hitAtk = \hitAtk.kr(0.02);
			var dry, sig, lat, wetSig, dir, lvlDry, lvlWet, rec, mean, std, walk, hits;
			if(src == 1) {
				// the recording itself, looped, brought to a level the model likes
				rec = Mix(PlayBuf.ar(~grBuf.numChannels, ~grBuf, BufRateScale.kr(~grBuf), loop: 1));
				rec = rec * (0.06 / Amplitude.kr(rec, 0.5, 1).max(0.002)).clip(0.05, 20).lag(0.5);
				dry = LPF.ar(rec.dup, cut);
			} {
				dry = LPF.ar(In.ar(~grBus.index, 2), cut);
			};
			sig = dry;
			if(name.notNil) {
				dir = \dir.kr(Array.fill(z, 0), 0.1);
				if(src == 2) {
					// no input: the latent vector wanders slowly around the mean of the recording
					mean = \lmean.kr(Array.fill(z, 0), 0.1);
					std = \lstd.kr(Array.fill(z, 1), 0.1);
					walk = Array.fill(z) { LFNoise2.kr(walkRate) };
					lat = K2A.ar(mean.clip(-3, 3) + (walk * std.clip(0.05, 1.5) * walkSize));   // limits for recordings the model does not know
				} {
					lat = NN(name, \encode).ar(dry.sum * 0.5 * inGain);
				};
				Out.kr(~grLatBus.index, A2K.kr(lat.keep(32)));
				lat = Lag.ar(lat, smooth);   // smoothing of the latent vector
				lat = lat + (dir * ((v * 2) - 1) * shift);
				Out.kr(~grLatBus.index + 32, A2K.kr(lat.keep(32)));
				wetSig = NN(name, \decode).ar(lat);
				// level compensation: the output level of a model can change a lot along a latent direction.
				// It is brought to the level of the input (model only: to a fixed level), so that the loudness
				// depends on the value only through the amp stage.
				lvlDry = if(src == 2) { DC.kr(0.06) } { Amplitude.kr(dry.sum * 0.5, 0.2, 1.0).max(1e-4) };
				lvlWet = Amplitude.kr(wetSig, 0.2, 1.0).max(1e-4);
				wetSig = wetSig * Select.kr(comp, [1, (lvlDry / lvlWet).clip(0.05, 20).lag(0.3)]);
				if(src == 2) {
					wetSig = LPF.ar(wetSig, cut);
					if(mode == 1) {   // percussive: hits at random times, faster with the value
						hits = EnvGen.kr(Env.perc(hitAtk, hitLen, 1, [2, -5]), Dust.kr(hitRate));
						wetSig = wetSig * hits;
					};
				};
				wetSig = wetSig.dup;
				sig = if(src == 2) { wetSig } { XFade2.ar(dry, wetSig, wet.linlin(0, 1, -1, 1)) };
			};
			sig = BHiShelf.ar(sig, 3500, 1.0, tame.neg);   // takes the edge off the highs
			Limiter.ar(sig * amp, 0.9)
		});
		if(name.notNil) { ~grSetDir.value; Ndef(\grOut).set(\shift, ~grShift, \comp, ~grComp) };
		if(src == 2) { ~grSetStats.value };
		Ndef(\grOut).set(\tame, ~grTame, \smooth, ~grSmooth, \walk, ~grWalk, \walkRate, ~grWalkRate);
		Ndef(\grOut).play;
		~grSet.(~grV);
	};
	// source "model only": mean and spread of the latents of the recording (rave_latent_stats.py)
	~grSetStats = {
		var path, rows, z = ~grZ, msg;
		if(~grFile.notNil and: { ~grModelIdx.notNil }) {
			path = ~grDir ++ "directions/" ++ ~grFiles[~grModelIdx].fileName.splitext[0] ++ "__" ++ ~grFile.basename.splitext[0] ++ ".stats.csv";
			if(File.exists(path)) {
				rows = CSVFileReader.read(path, true, true).collect { |r| r.collect { |x| x.asFloat } };
				if((rows.size >= 2) and: { rows[0].size == z }) {
					Ndef(\grOut).set(\lmean, rows[0], \lstd, rows[1]);
					msg = "model only: latents wander around the values of this recording";
				};
			};
		};
		if(msg.isNil) {
			Ndef(\grOut).set(\lmean, Array.fill(z, 0), \lstd, Array.fill(z, 1));
			msg = "model only: no latent statistics for this model and recording, using mean 0 and spread 1 (run rave_latent_stats.py)";
			if(~grModelIdx.notNil and: { ~grFile.notNil }) {
				"no statistics. In Terminal, in the RAVE_SC folder:  .venv/bin/python rave_latent_stats.py models/% audio/%".format(~grFiles[~grModelIdx].fileName, ~grFile.basename).postln;
			};
		};
		{ if(~grStatus.notNil) { ~grStatus.string = msg } }.defer;
	};
	~grNewDir = {
		var d = Array.fill(~grZ) { 1.0.rand2 };
		Ndef(\grOut).set(\dir, d / d.squared.mean.sqrt);
	};
	// computed direction: file written by rave_direction.py (one line, one number per latent dimension)
	~grDirFor = { |idx| ~grDir ++ "directions/" ++ ~grFiles[idx].fileName.splitext[0] ++ ".csv" };
	~grSetDir = {
		var path, d, ok = false, msg;
		if((~grDirMode == 0) and: { ~grModelIdx.notNil }) {
			path = ~grDirFor.(~grModelIdx);
			if(File.exists(path)) {
				d = CSVFileReader.read(path, true, true)[0].collect { |x| x.asFloat };
				if(d.size == ~grZ) { Ndef(\grOut).set(\dir, d); ok = true };
			};
		};
		if(ok.not) { ~grNewDir.value };
		msg = if(ok) { "latent direction: computed for this model (louder and brighter along it)" } {
			if(~grDirMode == 0) { "no direction file for this model: using a random direction (see the post window)" } { "random latent direction" }
		};
		if(ok.not and: { (~grDirMode == 0) and: { ~grModelIdx.notNil } }) {
			"no direction file. In Terminal, in the RAVE_SC folder:  .venv/bin/python rave_direction.py models/% audio/Seawaves1.wav".format(~grFiles[~grModelIdx].fileName).postln;
		};
		{ if(~grStatus.notNil) { ~grStatus.string = msg } }.defer;
	};
	// idx = nil: grains only. Otherwise the index in ~grFiles
	~grUseModel = { |idx|
		{
			var file, z;
			if(idx.isNil) {
				~grModelIdx = nil;
				~grBuild.value;
				if(~grPlayBtn.notNil and: { ~grPlayBtn.value == 0 }) { ~grStart.value };   // the grains are heard again
			} {
				file = ~grFiles[idx];
				z = ~grLatentSize.(file.fileName);
				if(z.isNil) {
					"Could not read the latent size from %".format(file.fileName).warn;
				} {
					if(~grLoaded.includes(("g" ++ idx).asSymbol).not) {
						NN.load(("g" ++ idx).asSymbol, file.fullPath);
						s.sync;
						~grLoaded.add(("g" ++ idx).asSymbol);
					};
					~grZ = z;
					~grModelIdx = idx;
					~grBuild.value;
					if(~grSrc != 0) { ~grStop.value };
					"RAVE model: % (latent size %)".format(file.fileName, z).postln;
				};
			};
		}.fork;
	};

	// ---- the value
	~grSet = { |val|
		var p;
		~grV = val.clip(0, 1);
		p = ~grParams.(~grMode, ~grV);
		Ndef(\grOut).set(\cut, p[\cut], \amp, (p[\db] + ~grMakeup[~grMode]).dbamp * ~grVol, \v, ~grV,
			\hitRate, p[\rate], \hitLen, p[\glen], \hitAtk, 0.004 + (~grSoft * 0.03));
		{
			if(~grValSl.notNil) { ~grValSl.value = ~grV };
			if(~grUv.notNil and: { ~grUv.isClosed.not }) { ~grUv.refresh };
			if(~grInfo.notNil) {
				~grInfo.string = "value %\n% grains per s | grain % ms | about % overlapping\nlow-pass % Hz | level % dB (relative)".format(
					~grV.round(0.01), p[\rate].round(0.1), (p[\glen] * 1000).round(1), p[\overlap].round(0.1),
					p[\cut].round(10), p[\db].round(0.1));
			};
		}.defer;
	};
	~grSweep = { |secs|
		if(~grSweepR.notNil) { ~grSweepR.stop; ~grSweepR = nil };
		if(secs.notNil) {
			~grSweepR = Routine({
				var steps = (secs / 0.05).asInteger.max(2);
				(2 * steps).do { |k|
					~grSet.(if(k < steps) { k / (steps - 1) } { 1 - ((k - steps) / (steps - 1)) });
					0.05.wait;
				};
				{ ~grSweepBtn.value = 0 }.defer;
			}).play(SystemClock);
		};
	};

	// ---- analysis table and recording
	~grAnalyse = {
		var key = ~grFile ++ ~grRegion;
		var cmd = "% % % --lengths %".format(~grQ.(~grPy), ~grQ.(~grScript), ~grQ.(~grFile), ~grRegion);
		{ ~grStatus.string = "analysing % ...".format(~grFile.basename) }.defer;
		"running: %".format(cmd).postln;
		cmd.unixCmd({ |code|
			if(code == 0 and: { ~grTried != key }) {
				~grTried = key;
				~grLoadTable.value;
			} {
				{ ~grStatus.string = "analysis failed or gave no usable table (exit code %), see the post window".format(code) }.defer;
			};
		});
	};
	~grLoadTable = {
		var path = ~grTable.value;
		var rows;
		if(File.exists(path).not) {
			~grAnalyse.value;
		} {
			rows = CSVFileReader.read(path, true, true).drop(1).collect { |r| r.collect { |x| x.asFloat } };
			if(rows.size < 20 or: { rows[0].size < 10 }) {
				~grAnalyse.value;
			} {
				~grTried = nil;
				~grRows = rows;
				~grRecent = [];
				~grPrepare.value;
				{ ~grStatus.string = "% regions from %".format(rows.size, ~grFile.basename) }.defer;
				~grStart.value;
				{ ~grPlayBtn.value = 0 }.defer;
			};
		};
	};
	~grOpen = { |path|
		{
			var sf = SoundFile.new;
			~grStop.value;
			~grRows = [];
			if(sf.openRead(path).not) {
				{ ~grStatus.string = "could not open " ++ path.basename }.defer;
			} {
				~grFile = path;
				~grSR = sf.sampleRate;
				~grCh = sf.numChannels;
				~grFrames = sf.numFrames;
				sf.close;
				if(~grBuf.notNil) { ~grBuf.free };
				~grBuf = Buffer.read(s, path);
				s.sync;
				if(~grModelIdx.notNil and: { ~grSrc != 0 }) { ~grBuild.value };   // the model source reads the buffer / its statistics
				~grLoadTable.value;
			};
		}.fork;
	};

	// ---- plot: regions of the recording, mapping curves, RAVE latent vector
	~grDraw = {
		var w = ~grUv.bounds.width, h = ~grUv.bounds.height;
		var v = ~grV, mode = ~grMode;
		var fnt = Font("Helvetica", 11);
		var txt = Color.grey(0.85), dim = Color.grey(0.45);
		var ax = 12, ay = 22, aw = w - 24, ah = (h * 0.46) - 30;
		var bx = 12, by = ay + ah + 28, bw = aw, bh = h * 0.2;
		var cx = 12, cy = by + bh + 44, cw = aw, ch = h - (by + bh + 44) - 10;
		var idx, key, width, near, now, p0, p1, pv, bars, z, scl;
		// A: all regions of the recording
		Pen.width = 1; Pen.color = dim; Pen.strokeRect(Rect(ax, ay, aw, ah));
		Pen.stringAtPoint("regions of the recording (0.5 s): right = brighter, up = burstier (both as ranks, so they fill the panel) | blue/pink: pool of this behaviour, yellow: can be picked now, rings: grains just played", Point(ax, 4), fnt, txt);
		if(~grPts.notNil and: { ~grRows.size > 0 and: { ~grPadIdx.notNil } }) {
			Pen.color = Color.grey(0.5, 0.35);
			~grPts.do { |pt| Pen.fillRect(Rect(ax + (pt[0] * aw) - 1, ay + ((1 - pt[1]) * ah) - 1, 2, 2)) };
			idx = if(mode == 0) { ~grPadIdx } { ~grPercIdx };
			key = if(mode == 0) { ~grPadKey } { ~grPercKey };
			width = if(mode == 0) { 0.15 } { 0.35 };
			Pen.color = if(mode == 0) { Color(0.3, 0.7, 0.9, 0.7) } { Color(0.9, 0.5, 0.8, 0.7) };
			idx.do { |i| var pt = ~grPts[i]; Pen.fillRect(Rect(ax + (pt[0] * aw) - 1.5, ay + ((1 - pt[1]) * ah) - 1.5, 3, 3)) };
			near = (0..(idx.size - 1)).select { |j| (key[j] - v).abs <= width };
			Pen.color = Color(1, 0.8, 0.2, 0.95);
			near.do { |j| var pt = ~grPts[idx[j]]; Pen.fillRect(Rect(ax + (pt[0] * aw) - 2.5, ay + ((1 - pt[1]) * ah) - 2.5, 5, 5)) };
			now = Main.elapsedTime;
			~grHist.do { |hh|
				var age = now - hh[1], pt = ~grPts[hh[0]];
				if(age < 2 and: { pt.notNil }) {
					Pen.color = Color(1, 1, 1, 1 - (age / 2));
					Pen.addArc(Point(ax + (pt[0] * aw), ay + ((1 - pt[1]) * ah)), 5 + (age * 9), 0, 2pi);
					Pen.stroke;
				};
			};
		};
		// B: what the value does
		Pen.color = dim; Pen.strokeRect(Rect(bx, by, bw, bh));
		Pen.stringAtPoint("what the value does to the grains (bar = position between the value-0 and value-1 setting)", Point(bx, by - 16), fnt, txt);
		p0 = ~grParams.(mode, 0); p1 = ~grParams.(mode, 1); pv = ~grParams.(mode, v);
		bars = [
			[\rate, Color(0.4, 0.8, 1), "grains per second", 1, 0.1, true],
			[\glen, Color(0.6, 1, 0.5), "grain length (ms)", 1000, 1, true],
			[\cut, Color(1, 0.7, 0.3), "low-pass cutoff (Hz)", 1, 10, true],
			[\db, Color(1, 0.4, 0.5), "level (dB, relative)", 1, 0.1, false]
		];
		bars.do { |br, k|
			var key = br[0], lo = p0[key], hi = p1[key], cur = pv[key];
			var norm = if((hi - lo).abs < 1e-6) { 0.5 } {
				if(br[5]) { (cur.log - lo.log) / (hi.log - lo.log) } { (cur - lo) / (hi - lo) }
			};
			var rowH = bh / 4, y = by + (k * rowH);
			Pen.color = Color.grey(0.2); Pen.fillRect(Rect(bx + 1, y + 3, bw - 2, rowH - 6));
			Pen.color = br[1].alpha_(0.8); Pen.fillRect(Rect(bx + 1, y + 3, (bw - 2) * norm.clip(0, 1), rowH - 6));
			Pen.stringAtPoint("% : now %   (value 0: %,  value 1: %)".format(br[2], (cur * br[3]).round(br[4]),
				(lo * br[3]).round(br[4]), (hi * br[3]).round(br[4])), Point(bx + 6, y + (rowH / 2) - 7), fnt, Color.white);
		};
		// C: the latent vector of the RAVE model
		Pen.color = dim; Pen.strokeRect(Rect(cx, cy, cw, ch));
		if(~grModelIdx.notNil and: { ~grLat.notNil }) {
			z = ~grZ.min(32);
			scl = max(1.5, ~grLat.keep(64).abs.maxItem);
			Pen.stringAtPoint("RAVE latent vector, one bar pair per dimension: grey = from the encoder, orange = after the shift by the value (equal to grey when 'latent shift' is 0)", Point(cx, cy - 16), fnt, txt);
			z.do { |d|
				var slotW = cw / z, mid = cy + (ch / 2);
				var b0 = (~grLat[d] / scl).clip(-1, 1) * (ch / 2), b1 = (~grLat[32 + d] / scl).clip(-1, 1) * (ch / 2);
				Pen.color = Color.grey(0.6);
				Pen.fillRect(Rect(cx + (d * slotW) + 2, mid.min(mid - b0), (slotW / 2) - 3, b0.abs.max(1)));
				Pen.color = Color(1, 0.6, 0.2);
				Pen.fillRect(Rect(cx + (d * slotW) + (slotW / 2), mid.min(mid - b1), (slotW / 2) - 3, b1.abs.max(1)));
			};
		} {
			Pen.stringAtPoint("RAVE latent vector: choose a RAVE model to see it", Point(cx, cy - 16), fnt, dim);
		};
	};

	// ---- window
	~grBuildWindow = {
		var win;
		if(~grWin.notNil and: { ~grWin.isClosed.not }) { ~grWin.close };
		~grStatus = StaticText().string_("");
		~grInfo = StaticText().string_("").minHeight_(60);
		~grValSl = Slider().orientation_(\horizontal).value_(~grV).minHeight_(36)
			.action_({ |sl| ~grSet.(sl.value) });
		~grFilePop = PopUpMenu().items_(~grFileList.collect { |f| f.basename })
			.action_({ |m| if(~grFileList.size > 0) { ~grOpen.(~grFileList[m.value]) } });
		~grRescanBtn = Button().states_([["rescan folder"]]).action_({
			var cur = ~grFile;
			~grFileList = ~grAudioFiles.value;
			if(cur.notNil and: { ~grFileList.includesEqual(cur).not }) { ~grFileList = ~grFileList ++ [cur] };
			~grFilePop.items = ~grFileList.collect { |f| f.basename };
			~grFilePop.value = ~grFileList.indexOfEqual(cur) ? 0;
		});
		~grChooseBtn = Button().states_([["choose another file..."]]).action_({
			FileDialog({ |paths|
				var path = paths[0];
				if(~grFileList.includesEqual(path).not) {
					~grFileList = ~grFileList ++ [path];
					~grFilePop.items = ~grFileList.collect { |f| f.basename };
				};
				~grFilePop.value = ~grFileList.indexOfEqual(path);
				~grOpen.(path);
			}, fileMode: 1, stripResult: true);
		});
		~grAnalyseBtn = Button().states_([["analyse again"]])
			.action_({ if(~grFile.notNil) { ~grTried = nil; ~grAnalyse.value } });
		~grModePop = PopUpMenu().items_([
			"pad: more present, brighter and denser with the value",
			"percussive: faster and denser with the value"]).action_({ |m|
				~grMode = m.value;
				~grRecent = [];
				if(~grModelIdx.notNil and: { ~grSrc == 2 }) { ~grBuild.value } { ~grSet.(~grV) };
		});
		~grModelPop = PopUpMenu().items_(["no RAVE (grains only)"] ++ ~grFiles.collect { |f| f.fileName })
			.action_({ |m| ~grUseModel.(if(m.value == 0) { nil } { m.value - 1 }) });
		~grSrcPop = PopUpMenu().items_([
			"source: the grain cloud (grains through RAVE)",
			"source: the recording itself (plain resynthesis)",
			"source: model only (no input, latents wander)"]).action_({ |m|
				var wasPlaying = ~grPlaying;
				~grSrc = m.value;
				~grStop.value;
				~grBuild.value;
				if(wasPlaying or: { ~grSrc == 0 }) { ~grStart.value };
		});
		~grWalkLabel = StaticText().string_("model only: size of the latent wander 0.5");
		~grWalkSl = Slider().orientation_(\horizontal).value_(~grWalk / 1.5).action_({ |sl|
			~grWalk = sl.value * 1.5;
			~grWalkLabel.string = "model only: size of the latent wander " ++ ~grWalk.round(0.01).asString ++ " (in spreads of the recording)";
			Ndef(\grOut).set(\walk, ~grWalk);
		});
		~grWalkRateSpec = ControlSpec(0.05, 2, \exp);
		~grWalkRateLabel = StaticText().string_("model only: speed of the wander 0.3 Hz");
		~grWalkRateSl = Slider().orientation_(\horizontal).value_(~grWalkRateSpec.unmap(~grWalkRate)).action_({ |sl|
			~grWalkRate = ~grWalkRateSpec.map(sl.value);
			~grWalkRateLabel.string = "model only: speed of the wander " ++ ~grWalkRate.round(0.01).asString ++ " Hz";
			Ndef(\grOut).set(\walkRate, ~grWalkRate);
		});
		~grWetSl = Slider().orientation_(\horizontal).value_(1)
			.action_({ |sl| Ndef(\grOut).set(\wet, sl.value) });
		~grInSpec = ControlSpec(0.1, 10, \exp);
		~grInSl = Slider().orientation_(\horizontal).value_(~grInSpec.unmap(1))
			.action_({ |sl| Ndef(\grOut).set(\inGain, ~grInSpec.map(sl.value)) });
		~grShiftLabel = StaticText().string_("latent shift: +-1 along the direction at value 0 and 1");
		~grShiftSl = Slider().orientation_(\horizontal).value_(~grShift / 2)
			.action_({ |sl|
				~grShift = sl.value * 2;
				~grShiftLabel.string = "latent shift: +-% along the direction at value 0 and 1".format(~grShift.round(0.01));
				Ndef(\grOut).set(\shift, ~grShift);
			});
		~grDirPop = PopUpMenu().items_(["direction: computed (louder and brighter)", "direction: random"])
			.action_({ |m| ~grDirMode = m.value; ~grSetDir.value });
		~grNewDirBtn = Button().states_([["new random direction"]]).action_({
			~grDirMode = 1; ~grDirPop.value = 1; ~grSetDir.value;
		});
		~grCompBtn = Button().states_([["level compensation: on"], ["level compensation: off"]])
			.action_({ |b| ~grComp = if(b.value == 0) { 1 } { 0 }; Ndef(\grOut).set(\comp, ~grComp) });
		~grSoftLabel = StaticText().string_("percussive attack: 19 ms");
		~grSoftSl = Slider().orientation_(\horizontal).value_(~grSoft).action_({ |sl|
			~grSoft = sl.value;
			~grSoftLabel.string = "percussive attack: " ++ ((0.004 + (~grSoft * 0.03)) * 1000).round(1).asString ++ " ms (softer = fewer clicks)";
		});
		~grCeilSpec = ControlSpec(3000, 16000, \exp);
		~grCeilLabel = StaticText().string_("brightest low-pass cutoff: 9000 Hz");
		~grCeilSl = Slider().orientation_(\horizontal).value_(~grCeilSpec.unmap(~grCeil)).action_({ |sl|
			~grCeil = ~grCeilSpec.map(sl.value);
			~grCeilLabel.string = "brightest low-pass cutoff: " ++ ~grCeil.round(10).asInteger.asString ++ " Hz";
			~grSet.(~grV);
		});
		~grTameLabel = StaticText().string_("take the edge off the highs: 4 dB");
		~grTameSl = Slider().orientation_(\horizontal).value_(~grTame / 12).action_({ |sl|
			~grTame = sl.value * 12;
			~grTameLabel.string = "take the edge off the highs: " ++ ~grTame.round(0.1).asString ++ " dB";
			Ndef(\grOut).set(\tame, ~grTame);
		});
		~grSmoothLabel = StaticText().string_("RAVE latent smoothing: 50 ms (0 to 300 ms)");
		~grSmoothSl = Slider().orientation_(\horizontal).value_(~grSmooth / 0.3).action_({ |sl|
			~grSmooth = sl.value * 0.3;
			~grSmoothLabel.string = "RAVE latent smoothing: " ++ (~grSmooth * 1000).round(1).asString ++ " ms";
			Ndef(\grOut).set(\smooth, ~grSmooth);
		});
		~grAvoidBtn = Button().states_([["skip the harshest regions: on"], ["skip the harshest regions: off"]])
			.action_({ |b| ~grAvoid = if(b.value == 0) { 1 } { 0 }; if(~grRows.size > 20) { ~grPrepare.value } });
		~grMulSpec = ControlSpec(0.25, 4, \exp);
		~grDensLabel = StaticText().string_("grain rate x 1");
		~grDensSl = Slider().orientation_(\horizontal).value_(~grMulSpec.unmap(1)).action_({ |sl|
			~grDens = ~grMulSpec.map(sl.value);
			~grDensLabel.string = "grain rate x " ++ ~grDens.round(0.01).asString;
			~grSet.(~grV);
		});
		~grLenLabel = StaticText().string_("grain length x 1");
		~grLenSl = Slider().orientation_(\horizontal).value_(~grMulSpec.unmap(1)).action_({ |sl|
			~grLen = ~grMulSpec.map(sl.value);
			~grLenLabel.string = "grain length x " ++ ~grLen.round(0.01).asString;
			~grSet.(~grV);
		});
		~grVolSpec = ControlSpec(0.05, 3, \exp);
		~grVolSl = Slider().orientation_(\horizontal).value_(~grVolSpec.unmap(~grVol))
			.action_({ |sl| ~grVol = ~grVolSpec.map(sl.value); ~grSet.(~grV) });
		~grSweepBox = NumberBox().value_(20).clipLo_(2).clipHi_(120).decimals_(0);
		~grSweepBtn = Button().states_([["sweep 0-1-0"], ["stop sweep"]])
			.action_({ |b| if(b.value == 1) { ~grSweep.(~grSweepBox.value) } { ~grSweep.(nil) } });
		~grPlayBtn = Button().states_([["stop"], ["play"]])
			.action_({ |b| if(b.value == 0) { ~grStart.value } { ~grStop.value } });

		~grUv = UserView().background_(Color.grey(0.12)).minWidth_(560).minHeight_(600)
			.drawFunc_({ ~grDraw.value });
		win = Window("Overlapping grains, one value", Rect(20, 30, 1560, 800));
		win.layout = HLayout([VLayout(
			StaticText().string_("recording"), ~grFilePop, HLayout(~grRescanBtn, ~grChooseBtn, ~grAnalyseBtn),
			StaticText().string_("behaviour"), ~grModePop,
			StaticText().string_("value (0 .. 1)"), ~grValSl, ~grInfo,
			~grDensLabel, ~grDensSl, ~grLenLabel, ~grLenSl,
			StaticText().string_("volume"), ~grVolSl,
			HLayout(~grSweepBtn, StaticText().string_("seconds"), ~grSweepBox),
			~grPlayBtn, ~grStatus, nil
		), stretch: 1], [VLayout(
			StaticText().string_("softer, less sharp"),
			~grSoftLabel, ~grSoftSl, ~grCeilLabel, ~grCeilSl, ~grTameLabel, ~grTameSl, ~grAvoidBtn,
			StaticText().string_("RAVE model"), ~grModelPop, ~grSrcPop, ~grWalkLabel, ~grWalkSl, ~grWalkRateLabel, ~grWalkRateSl,
			StaticText().string_("RAVE mix (0 = grains only, 1 = only the model)"), ~grWetSl,
			StaticText().string_("input gain to the model"), ~grInSl,
			~grShiftLabel, ~grShiftSl, ~grDirPop, ~grNewDirBtn, ~grCompBtn, ~grSmoothLabel, ~grSmoothSl, nil
		), stretch: 1], [~grUv, stretch: 2]);
		~grPoll = Routine({
			loop {
				if(~grModelIdx.notNil) { ~grLatBus.getn(64, { |vals| ~grLat = vals }) };
				if(~grUv.notNil and: { ~grUv.isClosed.not }) { ~grUv.refresh };
				0.1.wait;
			};
		}).play(AppClock);
		win.onClose = { ~grStop.value; ~grSweep.(nil); if(~grPoll.notNil) { ~grPoll.stop; ~grPoll = nil }; Ndef(\grOut).free(0.5) };
		win.front;
		~grWin = win;
		if(~grFileList.isEmpty) {
			~grStatus.string = "no recordings found in " ++ ~grAudioDir;
		} {
			~grFilePop.value = ~grFileList.detectIndex { |f| f.basename == "Seawaves1.wav" } ? 0;
			~grOpen.(~grFileList[~grFilePop.value]);
		};
	};

	~grBuild.value;
	{ ~grBuildWindow.value }.defer;
};
)
