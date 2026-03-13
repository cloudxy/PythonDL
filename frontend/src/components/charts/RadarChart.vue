<template>
  <div class="w-full h-full">
    <Radar v-if="loaded" :data="chartData" :options="chartOptions" />
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { Radar } from 'vue-chartjs'
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Title,
  Tooltip,
  Legend
} from 'chart.js'

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Title,
  Tooltip,
  Legend
)

const props = defineProps({
  title: {
    type: String,
    default: ''
  },
  labels: {
    type: Array,
    required: true
  },
  datasets: {
    type: Array,
    required: true
  },
  showLegend: {
    type: Boolean,
    default: true
  },
  showTooltip: {
    type: Boolean,
    default: true
  },
  smooth: {
    type: Boolean,
    default: true
  },
  height: {
    type: Number,
    default: 300
  }
})

const loaded = ref(false)

onMounted(() => {
  loaded.value = true
})

const chartData = computed(() => ({
  labels: props.labels,
  datasets: props.datasets.map((dataset, index) => ({
    ...dataset,
    fill: true,
    tension: props.smooth ? 0.4 : 0,
    backgroundColor: dataset.backgroundColor || getColor(index, 0.2),
    borderColor: dataset.borderColor || getColor(index),
    pointRadius: dataset.pointRadius ?? 3,
    pointHoverRadius: dataset.pointHoverRadius ?? 5,
    pointBackgroundColor: dataset.pointBackgroundColor || getColor(index),
    pointBorderColor: '#fff',
    pointBorderWidth: 2
  }))
}))

const chartOptions = computed(() => ({
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    title: {
      display: !!props.title,
      text: props.title,
      font: {
        size: 16,
        weight: 'bold'
      },
      padding: 20
    },
    legend: {
      display: props.showLegend,
      position: 'top',
      labels: {
        usePointStyle: true,
        pointStyle: 'circle'
      }
    },
    tooltip: {
      enabled: props.showTooltip,
      backgroundColor: 'rgba(0, 0, 0, 0.8)',
      titleFont: {
        size: 14
      },
      bodyFont: {
        size: 13
      },
      padding: 12,
      cornerRadius: 8
    }
  },
  scales: {
    r: {
      angleLines: {
        display: true,
        color: 'rgba(0, 0, 0, 0.1)'
      },
      grid: {
        color: 'rgba(0, 0, 0, 0.05)'
      },
      pointLabels: {
        font: {
          size: 12,
          weight: 'bold'
        },
        color: '#4b5563'
      },
      ticks: {
        backdropColor: 'transparent',
        color: '#6b7280',
        stepSize: 1
      }
    }
  }
}))

function getColor(index, alpha = 1) {
  const colors = [
    `rgba(59, 130, 246, ${alpha})`,
    `rgba(16, 185, 129, ${alpha})`,
    `rgba(245, 158, 11, ${alpha})`,
    `rgba(239, 68, 68, ${alpha})`,
    `rgba(139, 92, 246, ${alpha})`,
    `rgba(236, 72, 153, ${alpha})`
  ]
  return colors[index % colors.length]
}
</script>

<style scoped>
div {
  height: v-bind('height + "px"');
}
</style>
